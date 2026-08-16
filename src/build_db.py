"""Load the raw Olist CSVs into a relational SQLite database.

The tables stay separate and the real foreign keys are declared, so answering a
business question requires actual joins. Column names are preserved exactly as
they appear in the source files, including the misspellings `product_name_lenght`
and `product_description_lenght`, because the agent is meant to discover the real
schema rather than rely on names it might guess.

One relationship is deliberately not declared as a foreign key:
products.product_category_name references product_category_translation, but 13
product rows carry a category absent from the translation table and 610 rows are
null. Declaring it would force us to either drop real rows or disable enforcement,
so it stays a documented lookup relationship instead.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import RAW_FILES, Config
from src.profile_data import load_raw

SCHEMA = """
CREATE TABLE customers (
    customer_id              TEXT PRIMARY KEY,
    customer_unique_id       TEXT NOT NULL,
    customer_zip_code_prefix INTEGER NOT NULL,
    customer_city            TEXT NOT NULL,
    customer_state           TEXT NOT NULL
);

CREATE TABLE sellers (
    seller_id              TEXT PRIMARY KEY,
    seller_zip_code_prefix INTEGER NOT NULL,
    seller_city            TEXT NOT NULL,
    seller_state           TEXT NOT NULL
);

CREATE TABLE product_category_translation (
    product_category_name         TEXT PRIMARY KEY,
    product_category_name_english TEXT NOT NULL
);

CREATE TABLE products (
    product_id                 TEXT PRIMARY KEY,
    product_category_name      TEXT,
    product_name_lenght        INTEGER,
    product_description_lenght INTEGER,
    product_photos_qty         INTEGER,
    product_weight_g           REAL,
    product_length_cm          REAL,
    product_height_cm          REAL,
    product_width_cm           REAL
);

CREATE TABLE geolocation (
    geolocation_zip_code_prefix INTEGER NOT NULL,
    geolocation_lat             REAL NOT NULL,
    geolocation_lng             REAL NOT NULL,
    geolocation_city            TEXT NOT NULL,
    geolocation_state           TEXT NOT NULL
);

CREATE TABLE orders (
    order_id                      TEXT PRIMARY KEY,
    customer_id                   TEXT NOT NULL,
    order_status                  TEXT NOT NULL,
    order_purchase_timestamp      TEXT NOT NULL,
    order_approved_at             TEXT,
    order_delivered_carrier_date  TEXT,
    order_delivered_customer_date TEXT,
    order_estimated_delivery_date TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE TABLE order_items (
    order_id            TEXT NOT NULL,
    order_item_id       INTEGER NOT NULL,
    product_id          TEXT NOT NULL,
    seller_id           TEXT NOT NULL,
    shipping_limit_date TEXT NOT NULL,
    price               REAL NOT NULL,
    freight_value       REAL NOT NULL,
    PRIMARY KEY (order_id, order_item_id),
    FOREIGN KEY (order_id) REFERENCES orders (order_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id),
    FOREIGN KEY (seller_id) REFERENCES sellers (seller_id)
);

CREATE TABLE order_payments (
    order_id             TEXT NOT NULL,
    payment_sequential   INTEGER NOT NULL,
    payment_type         TEXT NOT NULL,
    payment_installments INTEGER NOT NULL,
    payment_value        REAL NOT NULL,
    PRIMARY KEY (order_id, payment_sequential),
    FOREIGN KEY (order_id) REFERENCES orders (order_id)
);

CREATE TABLE order_reviews (
    review_id               TEXT NOT NULL,
    order_id                TEXT NOT NULL,
    review_score            INTEGER NOT NULL,
    review_comment_title    TEXT,
    review_comment_message  TEXT,
    review_creation_date    TEXT NOT NULL,
    review_answer_timestamp TEXT NOT NULL,
    PRIMARY KEY (review_id, order_id),
    FOREIGN KEY (order_id) REFERENCES orders (order_id)
);

CREATE INDEX idx_orders_customer   ON orders (customer_id);
CREATE INDEX idx_orders_status     ON orders (order_status);
CREATE INDEX idx_orders_purchased  ON orders (order_purchase_timestamp);
CREATE INDEX idx_items_product     ON order_items (product_id);
CREATE INDEX idx_items_seller      ON order_items (seller_id);
CREATE INDEX idx_payments_order    ON order_payments (order_id);
CREATE INDEX idx_reviews_order     ON order_reviews (order_id);
CREATE INDEX idx_products_category ON products (product_category_name);
CREATE INDEX idx_geo_zip           ON geolocation (geolocation_zip_code_prefix);
"""

LOAD_ORDER = [
    "customers",
    "sellers",
    "product_category_translation",
    "products",
    "geolocation",
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
]

INTEGER_COLUMNS = {
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
}


def _rows_for_insert(frame: pd.DataFrame) -> list[tuple]:
    prepared = frame.copy()
    for column in prepared.columns:
        if column in INTEGER_COLUMNS:
            prepared[column] = prepared[column].astype("Int64")
    prepared = prepared.astype(object).where(pd.notna(prepared), None)
    return list(prepared.itertuples(index=False, name=None))


def build(config: Config, force: bool = False) -> dict:
    db_path: Path = config.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        if not force:
            raise FileExistsError(f"{db_path} already exists, pass force=True to rebuild")
        db_path.unlink()

    frames = load_raw(config)

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)

        loaded = {}
        for table in LOAD_ORDER:
            frame = frames[table]
            columns = list(frame.columns)
            placeholders = ", ".join("?" for _ in columns)
            statement = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            connection.executemany(statement, _rows_for_insert(frame))
            loaded[table] = len(frame)

        connection.commit()

        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"foreign key violations after load: {violations[:5]}")

        verified = {}
        for table in LOAD_ORDER:
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            verified[table] = count
            if count != loaded[table]:
                raise RuntimeError(f"{table}: inserted {loaded[table]} but database holds {count}")

        connection.execute("ANALYZE")
        connection.commit()
    finally:
        connection.close()

    return {
        "db_path": str(db_path),
        "size_bytes": db_path.stat().st_size,
        "row_counts": verified,
        "foreign_key_check": "clean",
        "source_files": {name: RAW_FILES[name] for name in LOAD_ORDER},
    }
