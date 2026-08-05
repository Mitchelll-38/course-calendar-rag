import { SearchWorkspace } from "./search-workspace";

export default function Home() {
  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="MacReg home">
          <span className="brand-mark">M</span>
          <span>MacReg</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#search">Search</a>
          <a href="#method">Method</a>
          <a href="http://localhost:8000/docs">API</a>
        </nav>
      </header>

      <section className="intro" id="top">
        <div>
          <p className="eyebrow">Academic calendar research</p>
          <h1>Ask the calendar.<br />Inspect the evidence.</h1>
          <p className="intro-copy">Find registration rules and course details with ranked passages you can verify.</p>
        </div>
        <aside className="method-note" id="method">
          <span>Retrieval method</span>
          <strong>BM25 + dense vectors</strong>
          <p>Rankings are fused before the best passages are returned with source metadata.</p>
        </aside>
      </section>

      <SearchWorkspace />

      <footer>
        <p>Prototype content is synthetic. Confirm decisions against McMaster&apos;s official academic calendar.</p>
        <a href="https://github.com/Mitchelll-38/course-calendar-rag">View source</a>
      </footer>
    </main>
  );
}
