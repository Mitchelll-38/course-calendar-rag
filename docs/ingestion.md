# Calendar ingestion

McMaster's Registrar describes the Undergraduate Calendar as the official source for degree, program, course, rule, and regulation information. The current calendar is hosted by Modern Campus at `academiccalendars.romcmaster.ca`.

## Access policy

The ingestion command is intentionally conservative:

- HTTPS requests are restricted to an explicit host allowlist.
- `robots.txt` must be available and permit the project's descriptive user agent.
- A delay of at least one second is enforced before content requests.
- HTTP errors, including access restrictions, stop the run. The client does not retry around a denial.
- Every passage retains its source URL, retrieval timestamp, section, and content hash.
- No live calendar content is checked into this repository by this pull request.

At development time, the current calendar endpoint returned HTTP 403 to automated retrieval. Obtain explicit permission or an approved export before running ingestion. Do not attempt to bypass the restriction.

## Usage after permission

```shell
pip install -e ".[ingest]"
calendar-ingest "https://academiccalendars.romcmaster.ca/APPROVED_PATH" --output data/raw/calendar.jsonl
```

Raw outputs are gitignored. Review the output's attribution and content before loading it into an index.
