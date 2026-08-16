"""The benchmark question set.

Twenty four business questions in three bands. `single_table` needs one table.
`join` needs at least two joined tables. `ambiguous` is phrased the way someone
in an operations meeting would actually say it, using words that appear nowhere
in the schema, so the model has to work out that "product line" means category,
"promised" means the estimated delivery date and "come back" means a repeated
customer_unique_id.

Two pairs are deliberate duplicates in substance: q20 repeats q09 and q21 repeats
q03, each restated in vague business language. Identical ground truth with
different wording isolates the effect of phrasing alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from src import ground_truth as gt


@dataclass(frozen=True)
class Question:
    """One benchmark question.

    `require` records what the question actually asks for, so grading only
    demands a figure when the wording asks for one. "Which product line brings in
    the most money" asks for a name, and an answer naming the right category is
    correct whether or not it volunteers the revenue.
    """

    id: str
    text: str
    category: str
    truth: Callable[[], dict]
    require: str = "numbers"
    aliases: dict[str, list[str]] = field(default_factory=dict)
    note: str = ""


CATEGORY_ALIASES = {
    "health_beauty": ["health_beauty", "beleza_saude", "health and beauty", "health & beauty"],
    "bed_bath_table": ["bed_bath_table", "cama_mesa_banho", "bed bath table", "bed, bath and table"],
    "office_furniture": ["office_furniture", "moveis_escritorio", "office furniture"],
}

STATE_ALIASES = {
    "SP": ["SP", "sao paulo", "são paulo"],
    "RR": ["RR", "roraima"],
}

QUESTIONS: list[Question] = [
    Question("q01", "How many orders are in the database in total?", "single_table", gt.q01_total_orders),
    Question("q02", "How many orders have actually been delivered to the customer?", "single_table", gt.q02_delivered_orders),
    Question("q03", "What is the average review score customers leave?", "single_table", gt.q03_average_review_score),
    Question("q04", "Which payment method do customers use most often, and on how many payments?", "single_table", gt.q04_top_payment_type,
             require="both", aliases={"credit_card": ["credit_card", "credit card"]}),
    Question("q05", "How many distinct sellers are on the platform?", "single_table", gt.q05_seller_count),
    Question("q06", "What is the total value of all payments received?", "single_table", gt.q06_total_payment_value),
    Question("q07", "Which customer state has the largest number of customers, and how many?", "single_table", gt.q07_top_customer_state,
             require="both", aliases=STATE_ALIASES),
    Question("q08", "What is the highest price paid for a single item?", "single_table", gt.q08_most_expensive_item),

    Question("q09", "Which product category generates the most revenue, and how much? Give the English category name.", "join", gt.q09_top_category_by_revenue,
             require="both", aliases=CATEGORY_ALIASES),
    Question("q10", "Which product category has sold the most individual items, and how many? Give the English category name.", "join", gt.q10_top_category_by_items_sold,
             require="both", aliases=CATEGORY_ALIASES),
    Question("q11", "Which single seller has generated the most revenue from item sales, and how much?", "join", gt.q11_top_seller_revenue, require="both"),
    Question("q12", "For orders containing at least one product in the beleza_saude category, what is the average review score?", "join", gt.q12_health_beauty_review_score),
    Question("q13", "How many delivered orders contained more than one item?", "join", gt.q13_delivered_multi_item_orders),
    Question("q14", "Which customer state has the highest average freight cost per item, and what is that average?", "join", gt.q14_state_highest_avg_freight,
             require="both", aliases=STATE_ALIASES),
    Question("q15", "What is the average number of items per order?", "join", gt.q15_average_items_per_order),
    Question("q16", "Among product categories with at least 100 reviews, which has the lowest average review score, and what is it? Give the English category name.", "join", gt.q16_worst_category_by_review,
             require="both", aliases=CATEGORY_ALIASES),
    Question("q17", "How much item revenue came from orders placed during 2017?", "join", gt.q17_revenue_2017),

    Question("q18", "On average, how many days pass between a customer placing an order and it arriving at their door?", "ambiguous", gt.q18_average_delivery_days),
    Question("q19", "What share of delivered orders turned up later than the date the customer was promised?", "ambiguous", gt.q19_late_delivery_share),
    Question("q20", "Which product line brings in the most money for us?", "ambiguous", gt.q20_top_product_line_money,
             require="strings", aliases=CATEGORY_ALIASES,
             note="same underlying answer as q09, restated in vague business language"),
    Question("q21", "Overall, how happy are customers with what they buy from us?", "ambiguous", gt.q21_overall_satisfaction,
             note="same underlying answer as q03, restated in vague business language"),
    Question("q22", "How many of our customers have come back and bought from us more than once?", "ambiguous", gt.q22_repeat_customers),
    Question("q23", "What do we typically charge a customer for shipping on an order?", "ambiguous", gt.q23_typical_shipping_charge),
    Question("q24", "Which part of the country do most of our sales come from?", "ambiguous", gt.q24_top_sales_region,
             require="strings", aliases=STATE_ALIASES),
]


def by_id(question_id: str) -> Question:
    for question in QUESTIONS:
        if question.id == question_id:
            return question
    raise KeyError(f"no benchmark question with id '{question_id}'")


def categories() -> dict[str, int]:
    counts: dict[str, int] = {}
    for question in QUESTIONS:
        counts[question.category] = counts.get(question.category, 0) + 1
    return counts
