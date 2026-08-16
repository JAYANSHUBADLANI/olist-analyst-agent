# Why a grounded agent is worth its overhead

## The person this is for

Ana runs operations for a mid-sized Brazilian marketplace. She is accountable for delivery
performance and seller quality, and she is in meetings where someone asks a question that nobody
can answer on the spot: which categories are dragging the marketplace's review scores down, how
often is it missing the delivery date it promised, is freight worse in some states than others.

Ana does not write SQL. Her options today are both bad. She can file a ticket with the data team
and get an answer in two or three days, by which point the meeting has moved on, or she can ask a
chatbot and get an answer in four seconds that she has no way to check. The second option is worse
than it looks, because a confident wrong number does not announce itself. If a language model tells
her that 23 percent of orders are late when the real figure is 8 percent, nothing in the answer
looks different from a correct one.

What Ana actually needs is not speed. It is an answer she can put in front of her VP without
personally verifying it first.

## What the two systems on offer actually do

The single-shot baseline in this repo is the ordinary text-to-SQL chatbot. It is handed the schema,
it writes one query, the query runs, and it phrases whatever came back. When that query is right,
the answer is right and it cost almost nothing. When the query errors, or quietly returns the wrong
shape, nothing catches it. The system has no second look and no notion that anything went wrong.

The agent does four things the baseline cannot:

1. **It looks before it writes.** It calls `list_tables` and `describe_table` and reads the real
   column names. This matters most on questions phrased in business language. Nothing in this schema
   is called "promised delivery date" or "product line" or "customers who came back", and a system
   working from the words alone has to guess the mapping.
2. **It reads what came back.** A failed query returns the real SQLite error, and the agent gets
   another attempt with that error in front of it.
3. **It checks its own arithmetic against reality.** Every number in the final answer is matched
   against values that actually appeared in tool output during that same run.
4. **It says when it cannot tell.** An answer that fails the grounding check is rewritten, and the
   rewrite is allowed to say that a figure could not be established.

## The trade, stated plainly

The agent is slower and more expensive per question, and the multiple is not small. The measured
figures are in [benchmark_report.md](../reports/benchmark_report.md) rather than repeated here, so
that this document cannot drift out of step with the last run.

The way to think about the trade is per-question cost against the cost of one wrong number reaching
a decision. At these prices, thousands of agent questions cost less than one analyst-day. If the
agent prevents one bad decision a quarter, or saves Ana one afternoon of waiting, it has paid for
itself many times over. The latency is real but it is being compared to the wrong baseline: the
honest comparison is not four seconds against thirty, it is thirty seconds against two days in the
data team's queue.

The case gets weaker in exactly two situations, and both are worth naming:

- **High-volume, repetitive, well-understood questions.** If the same twenty questions get asked
  every day, build twenty dashboards. The agent is for the long tail of questions nobody anticipated.
- **Questions where being wrong is cheap.** If a rough figure for a slide is all that is needed, the
  extra spend buys nothing.

## What this does not solve

The grounding guard verifies that numbers trace back to executed queries. It does not verify that
the query asked the right question. If the agent writes a query that runs cleanly but quietly
answers something subtly different, for example counting orders when it should have counted items,
or including cancelled orders in a delivery-time average, the number it returns is genuinely
grounded and the guard will pass it. The answer will be wrong and the system will be confident.

This is the honest boundary of the approach. The guard closes off invented numbers, which is the
most common and most dangerous failure mode of a language model doing analysis. It does not close
off flawed reasoning, and no amount of numeric verification would. For questions where a wrong
answer is genuinely expensive, the SQL still needs a human read, and the transcript exists so that
read takes a minute rather than an afternoon.
