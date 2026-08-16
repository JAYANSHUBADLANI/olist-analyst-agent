"""Numeric grounding guard.

After the agent writes its final answer in natural language, this module pulls
every number out of that answer and checks each one against the values that
actually came back from tool calls during the same run. The comparison is made
against the recorded tool payloads, never against the model's own account of what
it did.

What counts as grounded:

  direct        the number appears in a tool result as is
  rounded       the number is a tool result rounded to the precision stated
  from_question the number was given in the user's question
  row_count     the number equals how many rows a query returned
  derived_pct   the number is written as a percentage and equals a / b * 100 for
                two values that both appear in tool results

Anything else is reported as ungrounded and fails the guard. Percentages are the
only derivation allowed, because permitting arbitrary arithmetic between observed
values would make almost any number appear grounded and the check would stop
meaning anything.

The honest limit of this guard: it verifies that numbers trace back to executed
queries. It does not verify that the SQL asked the right question. A query that
runs successfully but answers something subtly different will still ground its
result, and the guard will pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.trace import GuardResult, RunTrace

DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")

MAGNITUDES = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
}

NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?)"
    r"\s*(%|percent|thousand|million|billion|mn|bn|k|m|b)?"
    r"(?![A-Za-z0-9_]|\.\d)",
    re.IGNORECASE,
)

LIST_MARKER_PATTERN = re.compile(r"^\s*\d{1,2}[.)]\s+", re.MULTILINE)
PERCENT_WORDS = ("percent", "share", "proportion", "rate")

MAX_DERIVATION_INPUTS = 200

# Models asked to restate a figure often reach for typeset scientific notation rather than the
# tool's own e+NN string. Rewriting it to the ASCII form up front means one number parser
# instead of two, and the repair loop stops handing back a figure the guard cannot read.
SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")

SCI_NOTATION_PATTERN = re.compile(
    r"(\d(?:[\d,]*)?(?:\.\d+)?)\s*[×xX*]\s*10\s*\^?\s*([⁻⁺+-]?[⁰¹²³⁴⁵⁶⁷⁸⁹\d]+)"
)

# A figure written as 16,008,900 claims precision to the hundred, not to the unit, so its
# trailing zeros widen the tolerance. Left unbounded that reasoning would hand 1,000,000 a
# tolerance of 500,000 and ground almost anything, so the inferred width is also capped at a
# fraction of the figure itself. The cap only ever binds on numbers written to one or two
# significant digits, which are exactly the ones a fabricated answer tends to use.
TRAILING_ZERO_TOLERANCE_CAP = 0.001


def normalise_scientific_notation(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        mantissa = match.group(1)
        exponent = match.group(2).translate(SUPERSCRIPTS)
        return f"{mantissa}e{exponent}"

    return SCI_NOTATION_PATTERN.sub(replace, text)


@dataclass(frozen=True)
class Claim:
    """A number as the answer actually wrote it.

    `tolerance` is derived from how precisely the number was stated, so a value
    written as "4.09" accepts anything that rounds to 4.09 and a value written as
    "1.26 million" accepts anything that rounds to 1.26 million. That is stricter
    than a blanket percentage tolerance for precise figures and fairer to figures
    the model deliberately rounded.
    """

    text: str
    value: float
    is_percentage: bool
    decimals: int
    tolerance: float


@dataclass
class Observed:
    numbers: set[float]
    row_counts: set[float]
    strings: set[str]

    @property
    def all_numbers(self) -> set[float]:
        return self.numbers | self.row_counts


def observed_from_trace(trace: RunTrace) -> Observed:
    numbers: set[float] = set()
    row_counts: set[float] = set()
    strings: set[str] = set()

    for step in trace.steps:
        if step.kind != "tool_call" or not step.ok:
            continue
        payload = step.raw_payload or {}

        if step.tool == "run_sql":
            row_counts.add(float(payload.get("row_count", 0)))
            for row in payload.get("rows", []):
                for cell in row:
                    _absorb(cell, numbers, strings)

        elif step.tool == "list_tables":
            for entry in payload.get("tables", []):
                numbers.add(float(entry.get("rows", 0)))
                strings.add(str(entry.get("table", "")).lower())

        elif step.tool == "describe_table":
            numbers.add(float(payload.get("rows", 0)))
            strings.add(str(payload.get("table", "")).lower())
            for column in payload.get("columns", []):
                strings.add(str(column.get("name", "")).lower())
                for sample in column.get("sample_values", []):
                    _absorb(sample, numbers, strings)

    return Observed(numbers=numbers, row_counts=row_counts, strings=strings)


def _absorb(cell: object, numbers: set[float], strings: set[str]) -> None:
    if cell is None or isinstance(cell, bool):
        return
    if isinstance(cell, (int, float)):
        numbers.add(float(cell))
        return
    text = str(cell)
    strings.add(text.lower())
    for claim in extract_claims(text):
        numbers.add(claim.value)


def _parse_number(raw: str) -> tuple[float, int] | None:
    """Return the value and the decimal places the number was stated to.

    Decimal places drive the tolerance, so an exponent has to shift them: a tool
    that displays 16008872.12 as 1.60089e+07 has stated the value to the nearest
    hundred, not to five decimal places. Reading the mantissa alone would demand
    a precision the notation never claimed and reject the value as ungrounded.
    """
    text = raw.replace(",", "").lower()
    try:
        value = float(text)
    except ValueError:
        return None

    mantissa, _, exponent = text.partition("e")
    if "." in mantissa:
        decimals = len(mantissa.split(".")[1])
    else:
        # No decimal point, so the trailing zeros carry the precision claim.
        digits = mantissa.lstrip("-+")
        stripped = digits.rstrip("0")
        decimals = -(len(digits) - len(stripped)) if stripped else 0
    if exponent:
        decimals -= int(exponent)
    return value, decimals


def extract_claims(answer: str) -> list[Claim]:
    cleaned = LIST_MARKER_PATTERN.sub("", answer)
    cleaned = DATE_PATTERN.sub(" ", cleaned)
    cleaned = normalise_scientific_notation(cleaned)

    claims: list[Claim] = []
    for match in NUMBER_PATTERN.finditer(cleaned):
        raw = match.group(1)
        parsed = _parse_number(raw)
        if parsed is None:
            continue
        value, decimals = parsed

        suffix = (match.group(2) or "").lower()
        scale = MAGNITUDES.get(suffix, 1.0)
        value *= scale

        window = cleaned[match.end(): match.end() + 25].lower()
        is_percentage = suffix in {"%", "percent"} or any(w in window for w in PERCENT_WORDS)

        tolerance = 0.5 * (10.0**-decimals) * scale
        if decimals < 0:
            tolerance = min(tolerance, abs(value) * TRAILING_ZERO_TOLERANCE_CAP)

        claims.append(
            Claim(
                text=match.group(0).strip(),
                value=value,
                is_percentage=is_percentage,
                decimals=decimals,
                tolerance=tolerance,
            )
        )
    return claims


def _matches(claim: Claim, candidate: float) -> bool:
    return abs(claim.value - candidate) <= claim.tolerance


def classify(claim: Claim, observed: Observed, question_values: set[float]) -> str | None:
    for candidate in observed.numbers:
        if _matches(claim, candidate):
            return "direct" if claim.value == candidate else "rounded"

    for candidate in observed.row_counts:
        if _matches(claim, candidate):
            return "row_count"

    for candidate in question_values:
        if _matches(claim, candidate):
            return "from_question"

    if claim.is_percentage:
        pool = sorted(observed.all_numbers)[:MAX_DERIVATION_INPUTS]
        for numerator in pool:
            for denominator in pool:
                if denominator == 0:
                    continue
                if _matches(claim, numerator / denominator * 100.0):
                    return "derived_pct"
    return None


def check_answer(answer: str, trace: RunTrace, question: str) -> GuardResult:
    observed = observed_from_trace(trace)
    question_values = {c.value for c in extract_claims(question)}
    claims = extract_claims(answer)

    if not claims:
        return GuardResult(
            checked=True,
            passed=True,
            notes="answer stated no numbers, nothing to verify",
        )

    grounded: list[str] = []
    ungrounded: list[str] = []
    for claim in claims:
        label = classify(claim, observed, question_values)
        if label is None:
            ungrounded.append(claim.text)
        else:
            grounded.append(f"{claim.text} ({label})")

    return GuardResult(
        checked=True,
        passed=not ungrounded,
        unverified_numbers=ungrounded,
        grounded_numbers=grounded,
        notes=(
            f"{len(grounded)} of {len(claims)} numbers traced to tool output"
            if ungrounded
            else f"all {len(claims)} numbers traced to tool output"
        ),
    )


REPAIR_INSTRUCTION = (
    "Your answer contains {count} number(s) that do not appear in any tool result from this run: {numbers}. "
    "Every figure you state must come from a query you actually ran. Either run another query to establish "
    "those figures, or rewrite the answer using only values that appeared in tool output. Do not restate an "
    "unverified number."
)


def repair_prompt(result: GuardResult) -> str:
    return REPAIR_INSTRUCTION.format(
        count=len(result.unverified_numbers),
        numbers=", ".join(result.unverified_numbers),
    )
