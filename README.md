# A grounded data-analyst agent over a real relational database

[![tests](https://github.com/JAYANSHUBADLANI/olist-analyst-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/JAYANSHUBADLANI/olist-analyst-agent/actions/workflows/tests.yml)

I built an agent that answers business questions about an e-commerce database it has never seen, by
actually querying it. It explores the schema itself, writes and runs real SQL, reads what comes
back, fixes its own broken queries, and then checks every number in its final answer against the
tool output from that same run before saying it out loud.

The database is the Brazilian e-commerce dataset published by Olist: nine tables, roughly 100,000
orders, real anonymised transactions with delivery dates, payment types, review scores and product
categories. The agent is not given the schema. It has to go and look.

## Why I built this rather than another RAG chatbot

Most LLM projects, including my own earlier ones, are a single-shot chain: retrieve some context,
stuff it into one prompt, get one answer. That pattern breaks in a specific and quiet way when the
question needs computation over real data. The model produces a fluent, confident, wrong number,
and nothing in the output distinguishes it from a right one.

I wanted to build the thing that pattern is missing: a loop that plans, calls tools, observes real
results, corrects itself when a step fails, and verifies its own output before returning it. The
interesting engineering is not the prompt. It is the tool design, the failure handling, and above
all the measurement.

## How it works

```
question
   |
   v
[ plan ]  the model states its approach
   |
   v
[ act ]   calls one of three read-only tools
   |         list_tables | describe_table | run_sql
   v
[ observe ] the real rows, or the real SQLite error, go back into the conversation
   |
   +--- needs another step? ---> back to act (capped)
   |
   v
[ answer ] natural language
   |
   v
[ guard ] every number matched against tool output from this run
   |
   +--- ungrounded number found? ---> one forced rewrite, then re-check
   |
   v
final answer + full transcript
```

The schema never appears in the agent's system prompt. That is the whole point: any correct answer
has to come from exploration the agent performed in that run, so the benchmark measures the loop
rather than the prompt.

### The tools

Three tools, all read-only, all scoped to one SQLite file.

| Tool | What it does |
| --- | --- |
| `list_tables` | Every table with its row count |
| `describe_table` | Columns, types, keys and sample values for one table |
| `run_sql` | Runs one SELECT and returns the real rows, or the real error |

Read-only is enforced in four independent layers rather than by asking the model nicely:

1. The connection is opened with SQLite's read-only URI flag.
2. The connection sets `PRAGMA query_only`.
3. The statement must parse as a single `SELECT` or `WITH`, with comments and string literals
   stripped before that check so `WHERE status = 'delete me'` is not mistaken for a mutation.
4. A keyword blocklist rejects `ATTACH`, `PRAGMA`, `load_extension`, `readfile` and the rest.

Every one of those layers has a test that tries to get past it.

### Self-correction

When a query fails, the agent receives the actual SQLite error text. It is not shielded from it and
it is not silently retried behind the scenes. This matters because the errors are informative:
`no such column: status` tells the model exactly what to fix, and the next attempt usually does.
Consecutive failures are capped so a confused run ends with an honest "I could not determine that"
rather than burning the step budget.

### The grounding guard

After the agent writes its answer, a separate pass extracts every number from that text and matches
each one against values that actually appeared in tool output during the run. The comparison is
against the recorded tool payloads, never against the model's own account of what it did.

A number is accepted if it appears in a result directly, is a sensible rounding of one, is the row
count of a query, came from the question itself, or is a percentage equal to `a / b * 100` for two
observed values. Percentages are the only derivation allowed. Permitting arbitrary arithmetic
between observed values would make almost anything look grounded and the check would stop meaning
anything.

Tolerance comes from how precisely the number was written, not from a blanket percentage. An answer
saying `4.09` accepts anything that rounds to 4.09; an answer saying `1.26 million` accepts anything
that rounds to 1.26 million. That is strict on precise claims and fair to deliberate rounding.

If anything fails, the agent gets one forced rewrite with the offending figures named, and both the
before and after are logged.

## The benchmark

24 questions in three bands:

- **single_table** (8), answerable from one table.
- **join** (9), needs at least two tables joined.
- **ambiguous** (7), phrased the way someone says it out loud. Nothing in this schema is called
  "product line", "the date the customer was promised", or "customers who came back". The model has
  to work out that these mean `product_category_name`, `order_estimated_delivery_date` and a
  repeated `customer_unique_id`.

Two pairs are deliberate duplicates in substance, restated in vague business language, so the effect
of phrasing alone is isolated: `q20` repeats `q09` and `q21` repeats `q03`.

**Ground truth is computed independently.** Every expected answer is calculated in pandas directly
over the raw CSV files, in `src/ground_truth.py`. It never touches the SQLite database and never
reuses a query the agent might write. Different engine, different code path, different source. If
ground truth came from the same SQL the agent produces, the benchmark would be grading the agent
against itself, which is not a benchmark.

**Grading is programmatic.** No model judges another model's answer. The answer text is scanned for
the values ground truth says must appear, within 2 percent relative tolerance so a correct answer
that rounds is credited. Grading only demands a figure when the question asked for one: "which
product line brings in the most money" asks for a name, so naming the right category is correct
whether or not the revenue is volunteered.

**The baseline is deliberately strong.** It is the ordinary text-to-SQL chain: full schema DDL handed
to it for free, one query, one shot, no retry, no verification. The agent never gets the schema. A
baseline crippled on purpose would make the comparison worthless, so this one is the best version of
the naive approach and the agent has to earn its margin.

## Results

Run against `google/gemini-2.5-flash-lite` via Vertex AI. Full figures in
[reports/benchmark_report.md](reports/benchmark_report.md), which this table is copied from verbatim.

| Metric | Baseline | Agent |
| --- | :---: | :---: |
| Task success rate | 83.3% (20/24) | 79.2% (19/24) |
| Avg tool calls per question | 1.0 | 4.0 |
| Guard trigger rate | 4.2% | 0.0% |
| Avg cost per question | $0.000126 | $0.000592 |
| Avg latency per question | 2.2s | 5.9s |

The agent loses to the naive baseline in this run, and I would rather say that plainly than bury it.
Every failure below was read from the actual transcript, not guessed from the score. Of the five,
one (`q11`) is the model consistently choosing to fold shipping cost into "revenue", a defensible
reading the ground truth just does not share, reproduced the same way across every run I made. The
rest are real reasoning misses, not measurement artifacts: a `HAVING COUNT() > 1` that returns one
row per qualifying order instead of the count of orders, a join on two columns that are not actually
the same key. The guard cannot catch any of these, on purpose. It checks that a number traces back to
a query result, not that the query answered the right question, see the limitation below.

**Results vary between runs, even with nothing else changed.** The questions are graded against a
live model call, not a scripted one. Two runs of literally identical code, back to back, landed on
79.2% and 75.0%. The same late-delivery share question is the clearest example: in one run the
agent's SQL correctly computed 7,826 late deliveries out of 96,470, an 8.11% share, and its prose
then stated "7.82%" anyway, a plain arithmetic slip. The grounding guard caught it, forced a rewrite,
the model repeated the same wrong figure, and the benchmark correctly scored it wrong, exactly the
failure mode this project exists to catch. In the other run the same question came back correct and
cleanly grounded on the first try, no guard intervention needed. I would rather show that spread than
quote one run as if it were exact. I also tried the benchmark against Groq's `openai/gpt-oss-120b`
earlier in development and got different figures again, which is expected and not the same
comparison: that is a genuinely different, larger model, not a data point for this system's own
run-to-run noise.

What is verified without an API key, deterministically, by the 103 tests in `tests/`:

- The database matches its source CSVs row for row, and SQLite and pandas agree on a join.
- All four read-only layers reject everything thrown at them.
- The agent loop recovers from a bad column name, and the real error text reaches the model.
- A model that states a plan without calling a tool gets nudged once rather than having that plan
  accepted as the final answer, a real gap live testing surfaced: one run's agent opened with a
  planning sentence and no tool call, which the loop previously accepted as a finished answer with
  the database never queried at all.
- The guard catches an invented figure, forces a rewrite, and passes the corrected answer.
- The step budget holds and token usage accumulates across turns.

Cost figures above are indicative, not verified: `config/pricing.json` carries the provider's public
per-token rate but its `retrieved` date is not stamped, since I have not independently reconfirmed it
against current pricing. Token counts themselves are exact, read from the API response's own usage
field.

## What this project actually demonstrates, and what it does not

Since this is a portfolio piece, the honest version.

**It does demonstrate:** tool design under a real security constraint, an explicit agent loop rather
than one function-calling turn dressed up as an agent, error handling that treats a failed tool call
as information rather than an exception, and an evaluation harness where the ground truth is
genuinely independent of the thing being measured. The measurement is the part I would defend
hardest. Plenty of agent demos assert that the agent is better; this one has a number, computed
against answers derived from a different engine, with a baseline that was given every advantage.

**It does not demonstrate, and I want to be explicit:**

- **The guard checks numeric grounding, not semantic correctness.** It verifies that numbers trace
  back to executed queries. It does not verify the SQL asked the right question. A query that runs
  cleanly but counts orders where it should have counted items produces a wrong answer whose numbers
  are perfectly grounded, and the guard will pass it. This is the single biggest limitation and no
  amount of numeric checking would fix it.
- **The agent loop is not free.** Real per-question cost and latency are measured and reported, not
  waved away. It makes several API calls where the baseline makes two.
- **24 questions is a small benchmark.** It is enough to separate two systems that differ a lot. It
  is not enough for confident claims about small differences. It is also small enough that live-model
  variance dominates: two runs of identical code landed on the same overall rate but disagreed on
  which specific questions failed, see Results above.
- **One database, one schema.** Nothing here shows the approach generalises to a warehouse with
  hundreds of tables, where schema exploration itself becomes the hard problem.
- **No fine-tuning, no multi-agent orchestration.** Both were deliberately out of scope. One
  well-measured agent is more defensible than an impressive-looking system nobody can evaluate.

## Setup

Four commands from a clean checkout to a finished benchmark.

```bash
pip install -r requirements.txt
python run.py 0
cp .env.example .env
python run.py 3
```

`run.py 0` downloads the nine Olist CSVs and checks each one against the sha256 recorded in
`config/data_manifest.json`. Kaggle is the canonical source but needs an authenticated account, so
the fetcher pulls the identical files from public mirrors instead and falls through to the next
mirror if one is unreachable or serves something that fails verification. A file whose hash does not
match is deleted rather than used. If you would rather use Kaggle directly:

```bash
kaggle datasets download -d olistbr/brazilian-ecommerce --unzip -p data/raw
```

Either way, `python run.py 1` profiles the files, builds the database and verifies it against them.
Phase 3 runs phase 1 automatically if the database is not there yet.

Set `GROQ_API_KEY` in `.env` before anything that calls the model. The client speaks plain
OpenAI-compatible HTTP, so pointing `LLM_BASE_URL` and `LLM_MODEL` at a different provider works
without code changes. Vertex AI is a tested alternative: set `GCP_PROJECT_ID` instead of a Groq key
and the client authenticates with a `gcloud auth print-access-token` fetched at run time rather than
a static key, see `.env.example`. Phases 0, 1 and 4 and the whole test suite need no key.

A provider that rate-limits by tokens per minute can throttle a full benchmark run to a crawl. The
client caps how long it will sleep on a single `retry-after` response so a throttled run stays
visibly retrying instead of looking hung with no output and no CPU use.

## Usage

```bash
python run.py 0                                  download the CSVs and verify their checksums
python run.py 1                                  profile the CSVs and build the database
python run.py 2                                  smoke test the loop on three questions
python run.py 3                                  full benchmark, baseline and agent
python run.py 4                                  rebuild the report from the last run
python run.py ask "which categories ship late?"  watch the agent answer one question
python run.py ask --baseline "same question"     watch the single-shot baseline instead
python -m pytest                                 103 tests, no API key needed
```

`ask` prints the plan, every tool call with its SQL, the real rows returned, any error and the retry
that followed, the final answer, the guard verdict, and the token and latency cost of the run.

## Repo layout

```
run.py                   entrypoint for every phase
config/pricing.json      token rates, used to turn measured token counts into dollars
config/data_manifest.json sha256 for every raw file, checked on download
src/
  config.py              paths, model settings, agent limits
  fetch_data.py          verified download of the Olist CSVs
  profile_data.py        empirical verification of the raw CSVs
  build_db.py            CSVs to SQLite, real foreign keys
  tools.py               the three read-only tools and the SQL validator
  llm.py                 OpenAI-compatible client, usage and cost accounting
  agent.py               the plan/act/observe loop
  baseline.py            single-shot comparison
  guard.py               numeric grounding guard
  benchmark.py           the 24 questions
  ground_truth.py        independent answers, computed in pandas
  evaluate.py            benchmark runner and programmatic grading
  report.py              results to markdown
  cli.py                 live trace rendering
tests/                   103 tests, including a scripted fake LLM for the loop
docs/business_case.md    who this is for and why the overhead is worth it
reports/                 data profile and benchmark output
```

## Notes on the data

I verified the schema empirically rather than trusting the dataset description, and two things are
worth knowing:

- `review_id` is **not** unique. 814 rows share an id with another row, so the primary key on
  `order_reviews` is the composite `(review_id, order_id)`.
- `products.product_category_name` has 610 nulls and 13 rows in two categories that are missing from
  the translation table entirely. I left that relationship undeclared rather than dropping real rows
  to satisfy a constraint.
- The source misspells two columns as `product_name_lenght` and `product_description_lenght`. I kept
  the misspellings, because the agent is supposed to discover real column names rather than guess
  plausible ones.

The full profile is in [reports/data_profile.md](reports/data_profile.md).

## License

MIT, see [LICENSE](LICENSE).
