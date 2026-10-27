# Calendar ingestion

McMaster's Registrar describes the Undergraduate Calendar as the official source for degree, program, course, rule, and regulation information. The current calendar is hosted by Modern Campus at `academiccalendars.romcmaster.ca`.

## Access policy

The ingestion command is intentionally conservative:

- HTTPS requests are restricted to an explicit host allowlist.
- Redirects are rejected for both robots and content requests.
- `robots.txt` must be available and permit the project's descriptive user agent.
- Content requests wait at least one second and honor any longer `robots.txt` crawl delay.
- HTTP errors, including access restrictions, stop the run. The client does not retry around a denial.
- Every passage retains its source URL, client UTC retrieval timestamp, section, and SHA-256 of its text.
- Empty extraction results fail before an output file is written.
- Existing output paths are rejected before fetching, and output creation is exclusive so an approved export cannot be overwritten by a concurrent run. Choose a new `--output` path for each run.
- No live calendar content is checked into this repository by this pull request.

At development time, the current calendar endpoint returned HTTP 403 to automated retrieval. Obtain explicit permission or an approved export before running ingestion. Do not attempt to bypass the restriction.

## Usage after permission

```shell
pip install -e ".[ingest]"
calendar-ingest "https://academiccalendars.romcmaster.ca/APPROVED_PATH" --output data/raw/calendar.jsonl
```

Raw outputs are gitignored. Review the output's attribution and content before loading it into an index.
