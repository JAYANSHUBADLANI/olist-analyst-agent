"""Independently computed answers for every benchmark question.

These are calculated with pandas directly over the raw CSV files. Nothing here
touches the SQLite database and nothing here reuses a query the agent might
write. Different engine, different code path, different source. That is the whole
point: if ground truth were produced by the same SQL the agent produces, the
benchmark would be grading the agent against itself.

Each function returns the value or values that a correct answer has to state.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.config import RAW_FILES, Config


@lru_cache(maxsize=1)
def _frames() -> dict[str, pd.DataFrame]:
    config = Config.load()
    frames = {name: pd.read_csv(config.raw_dir / filename) for name, filename in RAW_FILES.items()}

    orders = frames["orders"]
    for column in (
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ):
        orders[column] = pd.to_datetime(orders[column], errors="coerce")
    return frames


def _items_with_category() -> pd.DataFrame:
    frames = _frames()
    merged = frames["order_items"].merge(
        frames["products"][["product_id", "product_category_name"]], on="product_id", how="left"
    )
    return merged.merge(frames["product_category_translation"], on="product_category_name", how="left")


def q01_total_orders() -> dict:
    return {"numbers": [len(_frames()["orders"])]}


def q02_delivered_orders() -> dict:
    orders = _frames()["orders"]
    return {"numbers": [int((orders.order_status == "delivered").sum())]}


def q03_average_review_score() -> dict:
    return {"numbers": [float(_frames()["order_reviews"].review_score.mean())]}


def q04_top_payment_type() -> dict:
    counts = _frames()["order_payments"].payment_type.value_counts()
    return {"numbers": [int(counts.iloc[0])], "strings": [str(counts.index[0])]}


def q05_seller_count() -> dict:
    return {"numbers": [int(_frames()["sellers"].seller_id.nunique())]}


def q06_total_payment_value() -> dict:
    return {"numbers": [float(_frames()["order_payments"].payment_value.sum())]}


def q07_top_customer_state() -> dict:
    counts = _frames()["customers"].customer_state.value_counts()
    return {"numbers": [int(counts.iloc[0])], "strings": [str(counts.index[0])]}


def q08_most_expensive_item() -> dict:
    return {"numbers": [float(_frames()["order_items"].price.max())]}


def q09_top_category_by_revenue() -> dict:
    items = _items_with_category()
    revenue = items.groupby("product_category_name_english").price.sum().sort_values(ascending=False)
    return {
        "numbers": [float(revenue.iloc[0])],
        "strings": [str(revenue.index[0])],
        "aliases": {str(revenue.index[0]): ["health_beauty", "beleza_saude", "health and beauty", "health & beauty"]},
    }


def q10_top_category_by_items_sold() -> dict:
    items = _items_with_category()
    counts = items.product_category_name_english.value_counts()
    return {"numbers": [int(counts.iloc[0])], "strings": [str(counts.index[0])]}


def q11_top_seller_revenue() -> dict:
    revenue = _frames()["order_items"].groupby("seller_id").price.sum().sort_values(ascending=False)
    return {"numbers": [float(revenue.iloc[0])], "strings": [str(revenue.index[0])]}


def q12_health_beauty_review_score() -> dict:
    frames = _frames()
    items = _items_with_category()
    target_orders = set(items.loc[items.product_category_name == "beleza_saude", "order_id"])
    reviews = frames["order_reviews"]
    subset = reviews[reviews.order_id.isin(target_orders)]
    return {"numbers": [float(subset.review_score.mean())]}


def q13_delivered_multi_item_orders() -> dict:
    frames = _frames()
    delivered = set(frames["orders"].loc[frames["orders"].order_status == "delivered", "order_id"])
    per_order = frames["order_items"].groupby("order_id").size()
    multi = per_order[(per_order > 1) & (per_order.index.isin(delivered))]
    return {"numbers": [int(len(multi))]}


def q14_state_highest_avg_freight() -> dict:
    frames = _frames()
    merged = (
        frames["order_items"][["order_id", "freight_value"]]
        .merge(frames["orders"][["order_id", "customer_id"]], on="order_id")
        .merge(frames["customers"][["customer_id", "customer_state"]], on="customer_id")
    )
    averages = merged.groupby("customer_state").freight_value.mean().sort_values(ascending=False)
    return {"numbers": [float(averages.iloc[0])], "strings": [str(averages.index[0])]}


def q15_average_items_per_order() -> dict:
    items = _frames()["order_items"]
    return {"numbers": [float(len(items) / items.order_id.nunique())]}


def q16_worst_category_by_review() -> dict:
    frames = _frames()
    items = _items_with_category()
    order_category = items[["order_id", "product_category_name_english"]].drop_duplicates()
    joined = frames["order_reviews"][["order_id", "review_score"]].merge(order_category, on="order_id")
    grouped = joined.groupby("product_category_name_english").review_score.agg(["mean", "count"])
    eligible = grouped[grouped["count"] >= 100].sort_values("mean")
    return {"numbers": [float(eligible["mean"].iloc[0])], "strings": [str(eligible.index[0])]}


def q17_revenue_2017() -> dict:
    frames = _frames()
    orders = frames["orders"]
    in_2017 = set(orders.loc[orders.order_purchase_timestamp.dt.year == 2017, "order_id"])
    items = frames["order_items"]
    return {"numbers": [float(items.loc[items.order_id.isin(in_2017), "price"].sum())]}


def q18_average_delivery_days() -> dict:
    orders = _frames()["orders"]
    delivered = orders[orders.order_delivered_customer_date.notna()]
    days = (delivered.order_delivered_customer_date - delivered.order_purchase_timestamp).dt.total_seconds() / 86400
    return {"numbers": [float(days.mean())]}


def q19_late_delivery_share() -> dict:
    orders = _frames()["orders"]
    delivered = orders[orders.order_delivered_customer_date.notna()]
    late = delivered.order_delivered_customer_date > delivered.order_estimated_delivery_date
    return {"numbers": [float(late.sum() / len(delivered) * 100)], "secondary_numbers": [int(late.sum())]}


def q20_top_product_line_money() -> dict:
    return q09_top_category_by_revenue()


def q21_overall_satisfaction() -> dict:
    return q03_average_review_score()


def q22_repeat_customers() -> dict:
    frames = _frames()
    merged = frames["orders"][["order_id", "customer_id"]].merge(
        frames["customers"][["customer_id", "customer_unique_id"]], on="customer_id"
    )
    per_customer = merged.groupby("customer_unique_id").order_id.nunique()
    return {"numbers": [int((per_customer > 1).sum())]}


def q23_typical_shipping_charge() -> dict:
    freight_per_order = _frames()["order_items"].groupby("order_id").freight_value.sum()
    return {"numbers": [float(freight_per_order.mean())]}


def q24_top_sales_region() -> dict:
    frames = _frames()
    merged = frames["orders"][["order_id", "customer_id"]].merge(
        frames["customers"][["customer_id", "customer_state"]], on="customer_id"
    )
    counts = merged.customer_state.value_counts()
    return {"numbers": [int(counts.iloc[0])], "strings": [str(counts.index[0])]}
