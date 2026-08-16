# Progress

Last updated: 2026-08-14

## Done

**Phase 1, data and tools**

- All nine Olist CSVs downloaded and verified. Kaggle needs authentication and this environment has
  no Kaggle CLI or credentials, so the files came from a Hugging Face mirror instead. Integrity was
  checked rather than assumed: the four LFS-tracked files match byte-for-byte against two other
  independent mirrors by SHA-256, and the five smaller files match by git blob hash. All nine sizes
  agree across three mirrors.
- Every file profiled empirically in `src/profile_data.py`. Row counts, columns, dtypes, null counts,
  key uniqueness and foreign key coverage are read from the actual files, not from the dataset
  description. Output in `reports/data_profile.md`.
- SQLite database built at `data/olist.db` (184 MB) with nine separate tables, declared foreign keys
  and indexes. `PRAGMA foreign_key_check` comes back clean and every table's row count matches its
  source CSV.
- Tool layer in `src/tools.py`: `list_tables`, `describe_table`, `run_sql`. Read-only is enforced in
  four independent layers, not by prompt instruction.
- Single-shot baseline in `src/baseline.py`.

**Phase 2, the agent**

- Multi-step plan/act/observe loop in `src/agent.py` with a step cap and a full transcript.
- Self-correction: tool errors are returned to the model verbatim so it can fix the specific problem.
  Consecutive SQL failures are capped.
- Grounding guard in `src/guard.py` plus a one-shot repair path when it flags an answer.

**Phase 3, benchmark harness**

- 24 questions in `src/benchmark.py`: 8 single-table, 9 join, 7 ambiguously phrased.
- Ground truth in `src/ground_truth.py`, computed in pandas over the raw CSVs. No SQL, no database,
  no reuse of anything the agent would write.
- Programmatic grading in `src/evaluate.py`. No model judges another model's answer.

**Phase 4, interface and reporting**

- Trace CLI in `src/cli.py`, live step-by-step rendering.
- Cost and latency logged per question from real API usage counts.
- Report generator in `src/report.py`.

**Testing**

- 101 tests passing. The agent loop, the retry path and the guard repair path are all covered
  end to end with a scripted fake LLM, so they are verified without an API key and deterministically.

**Portability**

- `python run.py 0` fetches the nine CSVs and verifies each against the sha256 in
  `config/data_manifest.json`, falling through to another mirror on failure. This exists because
  Kaggle needs an authenticated account, which would otherwise make the project impossible to run
  on a fresh machine. Tested against all three mirrors, plus the corrupted-file and dead-mirror
  paths.
- The four commands that take a clean checkout to a finished benchmark are in the README's Setup
  section.

**Benchmark**

- The 24-question benchmark has been run against a live model, baseline and agent. Figures are in
  `reports/benchmark_report.md` and summarised in the README's Results section.

## Open items

- **Pricing rates are unverified.** `config/pricing.json` carries `"retrieved": "UNVERIFIED"`. Token
  counts are measured exactly from the API response usage object, but the dollar rates multiplying
  them are configuration I have not checked against Groq's published pricing. The cost figures in
  the report are therefore indicative, not measured, and the README says so. Stamp a date in that
  file before quoting a dollar figure anywhere.
- **A number written to one or two significant figures gets a wider tolerance in the guard.**
  `TRAILING_ZERO_TOLERANCE_CAP` bounds this at 0.1 percent of the figure, so a round fabrication
  cannot buy itself an arbitrarily wide window, but a number like 1,000,000 is still checked less
  strictly than 1,234,567. That is the honest reading of how precisely each was stated.

## Design decisions worth revisiting

- **Provider and model.** Defaults to Groq with `openai/gpt-oss-120b`. Change `LLM_MODEL` in `.env`.
  The client speaks plain OpenAI-compatible HTTP, so any provider with that API works.
- **The baseline gets the full schema, the agent does not.** Deliberate, so the comparison is not
  rigged in the agent's favour: the baseline is given every advantage that does not require a tool
  call.
- **`products.product_category_name` has no foreign key.** 13 rows carry a category missing from the
  translation table and 610 are null, so enforcing it would mean dropping real rows.
