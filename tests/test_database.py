"""The database matches the files it was built from.

Ground truth is computed in pandas over the CSVs while the agent queries SQLite.
That only works as independent verification if the two hold the same data, so
these tests compare the two directly.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.config import RAW_FILES, Config

EXPECTED_ROWS = {
    "customers": 99441,
    "geolocation": 1000163,
    "order_items": 112650,
    "order_payments": 103886,
    "order_reviews": 99224,
    "orders": 99441,
    "product_category_translation": 71,
    "products": 32951,
    "sellers": 3095,
}


@pytest.fixture(scope="module")
def connection(config: Config):
    if not config.db_path.exists():
        pytest.skip("database not built, run `python run.py 1` first")
    conn = sqlite3.connect(f"file:{config.db_path}?mode=ro", uri=True)
    yield conn
    conn.close()


@pytest.mark.parametrize("table,expected", sorted(EXPECTED_ROWS.items()))
def test_row_counts_match_the_source_files(connection, config: Config, table, expected):
    in_db = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    in_csv = len(pd.read_csv(config.raw_dir / RAW_FILES[table]))
    assert in_db == in_csv == expected


def test_no_foreign_key_violations(connection):
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_declared_foreign_keys_are_present(connection):
    fks = {
        (table, row[3], row[2])
        for table in EXPECTED_ROWS
        for row in connection.execute(f"PRAGMA foreign_key_list({table})")
    }
    assert ("orders", "customer_id", "customers") in fks
    assert ("order_items", "order_id", "orders") in fks
    assert ("order_items", "product_id", "products") in fks
    assert ("order_items", "seller_id", "sellers") in fks
    assert ("order_payments", "order_id", "orders") in fks
    assert ("order_reviews", "order_id", "orders") in fks


def test_source_column_misspellings_are_preserved(connection):
    columns = {row[1] for row in connection.execute("PRAGMA table_info(products)")}
    assert "product_name_lenght" in columns
    assert "product_description_lenght" in columns


def test_sqlite_and_pandas_agree_on_a_join(connection, config: Config):
    in_db = connection.execute(
        "SELECT COUNT(*) FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
        "WHERE c.customer_state = 'SP'"
    ).fetchone()[0]

    orders = pd.read_csv(config.raw_dir / RAW_FILES["orders"])
    customers = pd.read_csv(config.raw_dir / RAW_FILES["customers"])
    merged = orders.merge(customers, on="customer_id")
    assert in_db == int((merged.customer_state == "SP").sum())
