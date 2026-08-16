# Benchmark results

Model `google/gemini-2.5-flash-lite`, 24 questions, graded against ground truth computed independently in pandas from the raw CSVs.

## Baseline versus agent

| Metric | Single-shot baseline | Full agent | Difference |
| --- | --- | --- | --- |
| Task success rate | 83.3% | 79.2% | -4.1 pts |
| Questions correct | 20/24 | 19/24 | -1 |
| Avg tool calls per question | 1.0 | 3.96 | 4.0x |
| Retry rate | 4.2% | 8.3% |  |
| Guard trigger rate | 4.2% | 0.0% |  |
| Avg tokens per question | 1,045 | 5,164 | 4.9x |
| Avg cost per question | $0.000126 | $0.000592 | 4.7x |
| Avg latency per question | 2.2s | 5.9s | 2.6x |
| Median latency | 1.9s | 4.0s |  |

## Success rate by question type

| Question type | Count | Baseline | Agent |
| --- | ---: | ---: | ---: |
| single_table | 8 | 100.0% | 100.0% |
| join | 9 | 77.8% | 66.7% |
| ambiguous | 7 | 71.4% | 71.4% |

## Self-correction, taken from real transcripts

**q16**: Among product categories with at least 100 reviews, which has the lowest average review score, and what is it? Give the English category name.

Failed attempt:

```sql
SELECT
  pct.product_category_name_english,
  AVG(or.review_score) AS average_score
FROM products AS p
JOIN order_reviews AS or
  ON p.product_id = or.order_id
JOIN product_category_translation AS pct
  ON p.product_category_name = pct.product_category_name
GROUP BY
  pct.product_category_name_english
HAVING
  COUNT(or.review_id) >= 100
ORDER BY
  average_score
LIMIT 1;
```

Tool returned: `near "or": syntax error`

**q24**: Which part of the country do most of our sales come from?

Failed attempt:

```sql

WITH
  CustomerOrderLocations AS (
    SELECT
      o.order_id,
      g.geolocation_state
    FROM orders AS o
    JOIN customers AS c
      ON o.customer_id = c.customer_id
    JOIN geolocation AS g
      ON CAST(c.customer_zip_code_prefix AS TEXT) LIKE g.geolocation_zip_code_prefix || '%'
  )
SELECT
  geolocation_state,
  COUNT(order_id) AS order_count
FROM CustomerOrderLocations
GROUP BY
  geolocation_state
ORDER BY
  order_count DESC
LIMIT 1;

```

Tool returned: `query exceeded the 30s time limit and was cancelled`

Next attempt after reading that error:

```sql

WITH
  CustomerState AS (
    SELECT
      c.customer_id,
      g.geolocation_state
    FROM customers AS c
    JOIN geolocation AS g
      ON SUBSTR(c.customer_zip_code_prefix, 1, 3) = g.geolocation_zip_code_prefix
  )
SELECT
  cs.geolocation_state,
  COUNT(o.order_id) AS order_count
FROM orders AS o
JOIN CustomerState AS cs
  ON o.customer_id = cs.customer_id
GROUP BY
  cs.geolocation_state
ORDER BY
  order_count DESC
LIMIT 1;

```

## Every question

| id | type | baseline | agent | agent tool calls | agent retries | guard |
| --- | --- | :---: | :---: | ---: | ---: | :---: |
| q01 | single_table | pass | pass | 2 | 0 | clean |
| q02 | single_table | pass | pass | 3 | 0 | clean |
| q03 | single_table | pass | pass | 3 | 0 | clean |
| q04 | single_table | pass | pass | 3 | 0 | clean |
| q05 | single_table | pass | pass | 3 | 0 | clean |
| q06 | single_table | pass | pass | 3 | 0 | clean |
| q07 | single_table | pass | pass | 3 | 0 | clean |
| q08 | single_table | pass | pass | 3 | 0 | clean |
| q09 | join | pass | pass | 4 | 0 | clean |
| q10 | join | pass | pass | 5 | 0 | clean |
| q11 | join | pass | fail | 4 | 0 | clean |
| q12 | join | pass | pass | 5 | 0 | clean |
| q13 | join | pass | fail | 4 | 0 | clean |
| q14 | join | pass | pass | 4 | 0 | clean |
| q15 | join | fail | pass | 4 | 0 | clean |
| q16 | join | fail | fail | 7 | 3 | clean |
| q17 | join | pass | pass | 4 | 0 | clean |
| q18 | ambiguous | pass | pass | 4 | 0 | clean |
| q19 | ambiguous | pass | pass | 4 | 0 | clean |
| q20 | ambiguous | pass | pass | 4 | 0 | clean |
| q21 | ambiguous | pass | pass | 3 | 0 | clean |
| q22 | ambiguous | fail | fail | 4 | 0 | clean |
| q23 | ambiguous | fail | fail | 5 | 0 | clean |
| q24 | ambiguous | pass | pass | 7 | 1 | clean |

## Run totals

- **baseline**: 25,070 tokens across 48 API calls, $0.0030 total, 55s wall clock.
- **agent**: 123,939 tokens across 109 API calls, $0.0142 total, 142s wall clock.

Dollar figures are measured token counts multiplied by the rates in `config/pricing.json`. Token counts come from the API response; the rates are configuration you should verify against current provider pricing.
