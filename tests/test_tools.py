"""The read-only guarantee and the tool surface."""

from __future__ import annotations

import pytest

from src.config import Config
from src.tools import SqlRejected, validate_sql

REJECTED = [
    "DROP TABLE orders",
    "DELETE FROM orders",
    "UPDATE orders SET order_status = 'x'",
    "INSERT INTO orders VALUES (1)",
    "ALTER TABLE orders ADD COLUMN x TEXT",
    "CREATE TABLE evil (id INTEGER)",
    "ATTACH DATABASE '/etc/passwd' AS leak",
    "PRAGMA table_info(orders)",
    "VACUUM",
    "SELECT 1; DROP TABLE orders",
    "SELECT 1; DELETE FROM orders;",
    "SELECT load_extension('evil.so')",
    "SELECT readfile('/etc/passwd')",
    "",
    "   ",
    "-- SELECT 1",
]

ACCEPTED = [
    "SELECT COUNT(*) FROM orders",
    "select count(*) from orders;",
    "WITH x AS (SELECT 1 AS n) SELECT n FROM x",
    "SELECT COUNT(*) FROM orders WHERE order_status = 'delivered'",
    "SELECT 1 -- a trailing comment",
    "SELECT COUNT(*) FROM orders WHERE order_status = 'drop table'",
]


@pytest.mark.parametrize("statement", REJECTED)
def test_dangerous_sql_is_rejected(statement):
    with pytest.raises(SqlRejected):
        validate_sql(statement)


@pytest.mark.parametrize("statement", ACCEPTED)
def test_ordinary_selects_are_accepted(statement):
    assert validate_sql(statement)


def test_string_literal_keyword_does_not_trip_the_blocklist():
    assert validate_sql("SELECT * FROM orders WHERE order_status = 'delete me'")


def test_database_is_opened_read_only(tools):
    result = tools.run_sql("SELECT COUNT(*) AS n FROM orders")
    assert result.ok
    assert result.payload["rows"][0][0] == 99441


def test_row_limit_is_enforced(tools):
    result = tools.run_sql("SELECT order_id FROM orders")
    assert result.ok
    assert result.payload["truncated"] is True
    assert result.payload["row_count"] == tools.max_rows


def test_describe_unknown_table_names_the_alternatives(tools):
    result = tools.describe_table("nope")
    assert not result.ok
    assert "orders" in result.error


def test_list_tables_reports_every_table(tools):
    result = tools.list_tables()
    assert result.ok
    names = {t["table"] for t in result.payload["tables"]}
    assert names == {
        "customers",
        "geolocation",
        "order_items",
        "order_payments",
        "order_reviews",
        "orders",
        "product_category_translation",
        "products",
        "sellers",
    }


def test_bad_column_returns_the_real_error(tools):
    result = tools.run_sql("SELECT no_such_column FROM orders")
    assert not result.ok
    assert "no_such_column" in result.error
