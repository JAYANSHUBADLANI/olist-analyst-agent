"""The numeric grounding guard."""

from __future__ import annotations

from src.guard import check_answer, extract_claims
from src.trace import RunTrace


def trace_with(rows, columns=("n",), row_count=None):
    trace = RunTrace(question_id="t", question="q", mode="agent", model="m")
    trace.add(
        kind="tool_call",
        tool="run_sql",
        ok=True,
        content="",
        raw_payload={
            "sql": "SELECT 1",
            "columns": list(columns),
            "rows": rows,
            "row_count": len(rows) if row_count is None else row_count,
            "truncated": False,
        },
    )
    return trace


def test_number_straight_from_a_query_is_grounded():
    trace = trace_with([[96478]])
    result = check_answer("There were 96,478 delivered orders.", trace, "how many?")
    assert result.passed
    assert any("direct" in g for g in result.grounded_numbers)


def test_fabricated_number_is_caught():
    trace = trace_with([[96478]])
    result = check_answer("There were 96,478 delivered orders worth R$ 4,200,000.", trace, "how many?")
    assert not result.passed
    assert "4,200,000" in " ".join(result.unverified_numbers)


def test_sensible_rounding_is_grounded():
    trace = trace_with([[4.086419]])
    result = check_answer("The average review score is 4.09.", trace, "average score?")
    assert result.passed
    assert any("rounded" in g for g in result.grounded_numbers)


def test_percentage_derived_from_two_observed_values_is_allowed():
    trace = trace_with([[7827, 96476]], columns=("late", "total"))
    result = check_answer("About 8.11% of delivered orders were late.", trace, "what share?")
    assert result.passed
    assert any("derived_pct" in g for g in result.grounded_numbers)


def test_arbitrary_arithmetic_is_not_treated_as_grounded():
    trace = trace_with([[100, 250]], columns=("a", "b"))
    result = check_answer("The combined total is 350 units.", trace, "total?")
    assert not result.passed


def test_dates_are_not_mistaken_for_claims():
    trace = trace_with([["2017-05-01"]], columns=("d",))
    result = check_answer("The earliest order was on 2017-05-01.", trace, "when?")
    assert result.passed


def test_answer_without_numbers_passes_trivially():
    trace = trace_with([[1]])
    result = check_answer("Health and beauty is the strongest category.", trace, "which category?")
    assert result.passed
    assert "nothing to verify" in result.notes


def test_value_from_the_question_is_not_flagged():
    trace = trace_with([[42]])
    result = check_answer("Across the top 5 categories the leader has 42.", trace, "show the top 5 categories")
    assert result.passed


def test_claims_ignore_ordered_list_markers():
    claims = extract_claims("1. health_beauty\n2. watches_gifts")
    assert claims == []


def test_scientific_notation_reads_as_one_number():
    claims = extract_claims("Revenue was 1.60089e+07 reais.")
    assert [c.value for c in claims] == [16008900.0]


def test_scientific_notation_echoed_from_tool_output_is_grounded():
    trace = trace_with([[16008872.12]])
    result = check_answer("Total revenue was 1.60089e+07 reais.", trace, "what is total revenue?")
    assert result.passed


def test_scientific_notation_does_not_ground_a_wrong_magnitude():
    trace = trace_with([[16008872.12]])
    result = check_answer("Total revenue was 1.60089e+08 reais.", trace, "what is total revenue?")
    assert not result.passed


def test_typeset_scientific_notation_reads_as_one_number():
    for written in ("1.60089 × 10⁷", "1.60089 x 10^7", "1.60089*10⁷"):
        claims = extract_claims(f"Revenue was {written} reais.")
        assert [c.value for c in claims] == [16008900.0], written


def test_typeset_scientific_notation_is_grounded():
    trace = trace_with([[16008872.12]])
    result = check_answer("Total revenue was 1.60089 × 10⁷ reais.", trace, "what is total revenue?")
    assert result.passed


def test_trailing_zeros_state_a_coarser_precision():
    trace = trace_with([[16008872.12]])
    result = check_answer("Total revenue was 16,008,900 reais.", trace, "what is total revenue?")
    assert result.passed


def test_trailing_zero_tolerance_stays_bounded():
    """A round number must not buy itself a tolerance wide enough to ground anything."""
    claim = extract_claims("about 1,000,000 orders")[0]
    assert claim.tolerance == 1000.0

    trace = trace_with([[880000.0]])
    result = check_answer("There were about 1,000,000 orders.", trace, "how many orders?")
    assert not result.passed
