"""Grading behaviour and the independence of ground truth."""

from __future__ import annotations

import pytest

from src.benchmark import QUESTIONS, by_id
from src.evaluate import grade_answer


def test_every_question_has_computable_ground_truth():
    for question in QUESTIONS:
        truth = question.truth()
        assert truth.get("numbers") or truth.get("strings"), f"{question.id} produced no ground truth"


def test_benchmark_covers_all_three_bands():
    bands = {q.category for q in QUESTIONS}
    assert bands == {"single_table", "join", "ambiguous"}
    assert len(QUESTIONS) >= 20


def test_question_ids_are_unique():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_exact_answer_is_correct():
    grade = grade_answer(by_id("q02"), "96,478 orders were delivered to the customer.")
    assert grade.correct


def test_sensibly_rounded_answer_is_credited():
    grade = grade_answer(by_id("q09"), "health_beauty, at about R$ 1.26 million.")
    assert grade.correct


def test_wrong_figure_is_marked_incorrect():
    grade = grade_answer(by_id("q02"), "About 12,000 orders were delivered.")
    assert not grade.correct
    assert grade.missing_numbers


def test_missing_category_name_is_marked_incorrect():
    grade = grade_answer(by_id("q09"), "The leading category earned 1258681.34.")
    assert not grade.correct
    assert grade.missing_strings


def test_accented_state_name_is_accepted():
    grade = grade_answer(by_id("q24"), "Most sales come from São Paulo.")
    assert grade.correct


def test_name_only_answer_passes_when_the_question_only_asks_for_a_name():
    grade = grade_answer(by_id("q20"), "Health and beauty brings in the most.")
    assert grade.correct


def test_average_review_score_rejects_a_coarse_rounding():
    assert grade_answer(by_id("q03"), "The average review score is 4.09.").correct
    assert not grade_answer(by_id("q03"), "The average review score is 3.5.").correct


@pytest.mark.parametrize("question_id", [q.id for q in QUESTIONS])
def test_ground_truth_is_stable_across_calls(question_id):
    question = by_id(question_id)
    assert question.truth() == question.truth()
