import SearchWorkspace from './search-workspace'

export default function Home() {
  return (
    <div className="site-shell">
      <header className="site-header">
        <div className="brand" aria-label="MacReg">
          <span className="brand-mark" aria-hidden="true">M</span>
          <span className="brand-name">MacReg<span className="brand-dot">.</span></span>
        </div>
        <span className="header-tag">ACADEMIC CALENDAR RESEARCH</span>
        <span className="preview-tag"><span className="status-dot" /> Research preview</span>
      </header>

      <main>
        <section className="intro" aria-labelledby="page-title">
          <div className="intro-copy">
            <p className="eyebrow"><span className="eyebrow-line" /> KNOW THE RULES, FIND YOUR PATH</p>
            <h1 id="page-title">The calendar, <em>in context.</em></h1>
            <p className="intro-lead">
              Ask a question about academic regulations or courses. MacReg shows ranked passages
              from the sample calendar and their source details.
            </p>
          </div>
          <div className="intro-aside" aria-label="How this research preview works">
            <span className="aside-number">01 / 03</span>
            <span className="aside-rule" />
            <p>Ask a question.<br />Explore the evidence.<br />Check the source.</p>
          </div>
        </section>

        <section className="workspace-section" aria-labelledby="workspace-title">
          <div className="section-heading">
            <div>
              <p className="section-kicker">EXPLORE THE CALENDAR</p>
              <h2 id="workspace-title">Start with a question.</h2>
            </div>
            <p className="section-note">Built to make the path from question to passage easier to follow.</p>
          </div>
          <SearchWorkspace />
        </section>

        <section className="method-note" aria-label="About this research preview">
          <span className="method-symbol" aria-hidden="true">i</span>
          <p><strong>A research preview, not academic advice.</strong> This demo uses a small synthetic
            dataset, not the live McMaster calendar. Verify policies and course requirements with
            official university sources before making decisions.</p>
        </section>
      </main>

      <footer className="site-footer">
        <span className="footer-brand">MacReg<span>.</span></span>
        <span>Academic calendar research prototype</span>
        <span>Synthetic sample content · For demonstration only</span>
      </footer>
    </div>
  )
}
