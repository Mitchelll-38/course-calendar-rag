'use client'

import { useEffect, useRef, useState, type FormEvent } from 'react'

type Retriever = 'bm25' | 'dense' | 'hybrid' | 'reranked'

type Source = {
  passage_id: string
  title: string
  section: string
  source: string
  score: number
  excerpt: string
}

type AnswerResponse = {
  question: string
  retriever: Retriever
  answer: string
  sources: Source[]
  disclaimer: string
}

const modes: { value: Retriever; label: string; description: string }[] = [
  { value: 'bm25', label: 'BM25', description: 'Keyword match' },
  { value: 'dense', label: 'Dense', description: 'Meaning match' },
  { value: 'hybrid', label: 'Hybrid', description: 'Combined search' },
  { value: 'reranked', label: 'Reranked', description: 'Refined order' },
]

const examples = [
  'Can I take COMP SCI 2C03?',
  'How do reserved seats work?',
  'What is the maximum course load?',
]

function safeSourceUrl(source: string): string | null {
  try {
    const url = new URL(source)
    return url.protocol === 'http:' || url.protocol === 'https:' ? url.href : null
  } catch {
    return null
  }
}

export default function SearchWorkspace() {
  const [question, setQuestion] = useState('')
  const [retriever, setRetriever] = useState<Retriever>('hybrid')
  const [answer, setAnswer] = useState<AnswerResponse | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const activeRequest = useRef<AbortController | null>(null)
  const requestNumber = useRef(0)

  useEffect(() => () => activeRequest.current?.abort(), [])

  function invalidateResults() {
    activeRequest.current?.abort()
    activeRequest.current = null
    requestNumber.current += 1
    setAnswer(null)
    setLoading(false)
    setError('')
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    activeRequest.current?.abort()
    const currentRequest = ++requestNumber.current
    setAnswer(null)
    setError('')
    setLoading(false)

    const trimmed = question.trim()
    if (trimmed.length < 3) {
      setError('Enter a question with at least 3 characters.')
      return
    }
    if (trimmed.length > 500) {
      setError('Keep your question within 500 characters.')
      return
    }

    const controller = new AbortController()
    activeRequest.current = controller
    setLoading(true)
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
      const response = await fetch(`${baseUrl.replace(/\/$/, '')}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed, retriever, top_k: 5 }),
        signal: controller.signal,
      })
      if (!response.ok) {
        throw new Error(
          response.status === 422
            ? 'The question could not be accepted. Check its length and try again.'
            : `The API returned an error (${response.status}). Please try again.`,
        )
      }
      const data = (await response.json()) as AnswerResponse
      if (currentRequest === requestNumber.current) setAnswer(data)
    } catch (cause) {
      if (currentRequest === requestNumber.current && !controller.signal.aborted) {
        setError(
          cause instanceof Error && cause.message !== 'Failed to fetch'
            ? cause.message
            : 'Could not reach the answer service. Check that the local API is running and try again.',
        )
      }
    } finally {
      if (currentRequest === requestNumber.current) {
        activeRequest.current = null
        setLoading(false)
      }
    }
  }

  return (
    <section className="workspace" aria-label="Course calendar search">
      <div className="query-panel">
        <form onSubmit={submit}>
          <label htmlFor="calendar-question">Ask a question about the calendar</label>
          <textarea
            id="calendar-question"
            className="query-input"
            value={question}
            onChange={(event) => {
              invalidateResults()
              setQuestion(event.target.value)
            }}
            maxLength={500}
            rows={3}
            placeholder="e.g. Can I take COMP SCI 2C03?"
            aria-describedby={error ? 'query-error' : undefined}
            aria-invalid={Boolean(error)}
          />
          {error && <p className="form-error" id="query-error" role="alert">{error}</p>}
          <fieldset className="mode">
            <legend>Search mode</legend>
            {modes.map((mode) => (
              <label key={mode.value}>
                <input
                  type="radio"
                  name="retriever"
                  value={mode.value}
                  checked={retriever === mode.value}
                  onChange={() => {
                    invalidateResults()
                    setRetriever(mode.value)
                  }}
                />
                <span>{mode.label}</span>
                <small>{mode.description}</small>
              </label>
            ))}
          </fieldset>
          <button type="submit" disabled={loading}>
            {loading ? 'Searching…' : 'Find an answer'}
          </button>
        </form>
        <div className="examples">
          <p>Try a sample question</p>
          {examples.map((example) => (
            <button key={example} type="button" onClick={() => {
              invalidateResults()
              setQuestion(example)
            }}>
              {example}
            </button>
          ))}
        </div>
      </div>

      <section className="results" aria-label="Search results" aria-live="polite">
        {loading ? (
          <p className="loading" role="status">Searching the sample calendar…</p>
        ) : answer ? (
          answer.sources.length === 0 ? (
            <p className="empty">No matching passages were found. Try another question or search mode.</p>
          ) : (
            <>
              <div className="answer">
                <p className="answer-label">Answer from the sample calendar</p>
                <p>{answer.answer}</p>
                <p className="disclaimer">{answer.disclaimer}</p>
              </div>
              <div className="sources">
                <h2>Sources</h2>
                <ol role="list">
                  {answer.sources.map((source, index) => {
                    const url = safeSourceUrl(source.source)
                    return (
                      <li key={`${source.passage_id}-${index}`}>
                        <span className="source-index" aria-label={`Source ${index + 1}`}>
                          {String(index + 1).padStart(2, '0')}
                        </span>
                        <div>
                          <h3>{source.title}</h3>
                          <p>{source.section}</p>
                          <p>{source.excerpt}</p>
                          <p>
                            Source: {url ? (
                              <a href={url} target="_blank" rel="noopener noreferrer">{source.source}</a>
                            ) : (
                              <span>{source.source}</span>
                            )}
                          </p>
                        </div>
                      </li>
                    )
                  })}
                </ol>
              </div>
            </>
          )
        ) : (
          <p className="empty">Ask a question to see an answer and its supporting passages.</p>
        )}
      </section>
    </section>
  )
}
