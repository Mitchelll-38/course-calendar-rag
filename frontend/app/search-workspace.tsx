"use client";

import { ArrowUpRight, BookOpenText, MagnifyingGlass } from "@phosphor-icons/react";
import { FormEvent, useState } from "react";

type Source = {
  passage_id: string;
  title: string;
  section: string;
  source: string;
  score: number;
  excerpt: string;
};

type Answer = { answer: string; sources: Source[]; disclaimer: string };

const examples = ["Can I take more than 30 units?", "What does antirequisite mean?", "COMP SCI 2C03 prerequisite"];

export function SearchWorkspace() {
  const [question, setQuestion] = useState("");
  const [retriever, setRetriever] = useState("hybrid");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (question.trim().length < 3) {
      setError("Enter at least three characters.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, retriever, top_k: 5 }),
      });
      if (!response.ok) throw new Error("Search service unavailable");
      setAnswer(await response.json());
    } catch {
      setError("The API could not be reached. Start FastAPI on port 8000 and try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="workspace" id="search">
      <div className="query-panel">
        <form onSubmit={submit}>
          <label htmlFor="question">Your question</label>
          <div className="query-input">
            <MagnifyingGlass size={22} weight="regular" aria-hidden="true" />
            <input id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about registration, course codes, or academic rules" />
            <button type="submit" disabled={loading}>{loading ? "Searching" : "Search"}</button>
          </div>
          {error && <p className="form-error" role="alert">{error}</p>}
          <fieldset>
            <legend>Retrieval mode</legend>
            {['bm25', 'dense', 'hybrid', 'reranked'].map((mode) => (
              <label className="mode" key={mode}>
                <input type="radio" name="retriever" value={mode} checked={retriever === mode} onChange={() => setRetriever(mode)} />
                {mode}
              </label>
            ))}
          </fieldset>
        </form>
        <div className="examples">
          <span>Try a question</span>
          {examples.map((example) => <button key={example} onClick={() => setQuestion(example)}>{example}</button>)}
        </div>
      </div>

      <div className="results" aria-live="polite">
        {loading ? <LoadingState /> : answer ? <AnswerView answer={answer} /> : <EmptyState />}
      </div>
    </section>
  );
}

function EmptyState() {
  return <div className="empty"><BookOpenText size={34} aria-hidden="true" /><h2>Your evidence will appear here</h2><p>Ask a specific question to see an answer and its ranked calendar passages.</p></div>;
}

function LoadingState() {
  return <div className="loading" aria-label="Searching"><span /><span /><span /></div>;
}

function AnswerView({ answer }: { answer: Answer }) {
  return <div className="answer"><span className="answer-label">Grounded answer</span><h2>{answer.answer}</h2><p className="disclaimer">{answer.disclaimer}</p><div className="sources"><h3>Ranked sources</h3>{answer.sources.map((source, index) => <article key={source.passage_id}><div className="source-index">{String(index + 1).padStart(2, '0')}</div><div><h4>{source.title}</h4><p>{source.excerpt}</p><span>{source.section}</span></div><a href={source.source} aria-label={`Open source for ${source.title}`}><ArrowUpRight size={18} aria-hidden="true" /></a></article>)}</div></div>;
}
