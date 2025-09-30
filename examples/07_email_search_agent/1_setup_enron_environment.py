# ============================================================
# NOTE: Before running this script, please create a folder:
#        TOOLBRAIN/data
#
# This directory is required to store datasets and outputs.
#
# 1. Environment Setup (run only once):
# This script will automatically download the Enron dataset 
# and create the file enron_emails.db
# python -m examples.07_email_search_agent.1_setup_enron_environment
# ============================================================

import sqlite3
import os
import logging
from datasets import load_dataset, Dataset, Features, Value, Sequence
from tqdm import tqdm
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "..", "..", "data", "enron_emails.db")

REPO_ID = "corbt/enron-emails"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


SQL_CREATE_TABLES = """
DROP TABLE IF EXISTS recipients;
DROP TABLE IF EXISTS emails_fts;
DROP TABLE IF EXISTS emails;

CREATE TABLE emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT UNIQUE,
    subject TEXT,
    from_address TEXT,
    date TEXT,
    body TEXT,
    file_name TEXT
);

CREATE TABLE recipients (
    email_id INTEGER,
    recipient_address TEXT,
    recipient_type TEXT, -- 'to', 'cc', 'bcc'
    FOREIGN KEY(email_id) REFERENCES emails(id) ON DELETE CASCADE
);
"""

SQL_CREATE_INDEXES_TRIGGERS = """
CREATE INDEX idx_emails_from ON emails(from_address);
CREATE INDEX idx_emails_date ON emails(date);
CREATE INDEX idx_emails_message_id ON emails(message_id);
CREATE INDEX idx_recipients_address ON recipients(recipient_address);

CREATE VIRTUAL TABLE emails_fts USING fts5(
    subject,
    body,
    content='emails',
    content_rowid='id'
);

CREATE TRIGGER emails_ai AFTER INSERT ON emails BEGIN
    INSERT INTO emails_fts (rowid, subject, body)
    VALUES (new.id, new.subject, new.body);
END;

CREATE TRIGGER emails_ad AFTER DELETE ON emails BEGIN
    DELETE FROM emails_fts WHERE rowid=old.id;
END;

CREATE TRIGGER emails_au AFTER UPDATE ON emails BEGIN
    UPDATE emails_fts SET subject=new.subject, body=new.body WHERE rowid=old.id;
END;

INSERT INTO emails_fts (rowid, subject, body) SELECT id, subject, body FROM emails;
"""

def download_dataset(repo_id: str) -> Dataset:
    logging.info(f"Loading dataset from Hugging Face Hub: {repo_id}")
    expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        }
    )
    dataset_obj = load_dataset(repo_id, features=expected_features, split="train")
    if not isinstance(dataset_obj, Dataset):
        raise TypeError(f"Error: Expected data type is Dataset, but received {type(dataset_obj)}")
    logging.info(
        f"Successfully loaded dataset '{repo_id}' with {len(dataset_obj)} records."
    )
    return dataset_obj


def create_database(db_path: str):
    """Create SQLite database and tables."""
    logging.info(f"Creating SQLite database and tables at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()
    conn.close()
    logging.info("Successfully created tables in the database.")


def populate_database(db_path: str, dataset: Dataset):
    """Load data from Hugging Face dataset into the database."""
    logging.info(f"Loading data into database {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")

    record_count = 0
    
    conn.execute("BEGIN TRANSACTION;")

    for email_data in tqdm(dataset, desc="Loading emails into DB"):
        assert isinstance(email_data, dict)

        # Extract and clean data
        date_obj: datetime = email_data["date"]
        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S") if date_obj else None
        
        to_list = [str(addr) for addr in email_data.get("to", []) if addr]
        cc_list = [str(addr) for addr in email_data.get("cc", []) if addr]
        bcc_list = [str(addr) for addr in email_data.get("bcc", []) if addr]

        # Insert into 'emails' table
        cursor.execute(
            """
            INSERT INTO emails (message_id, subject, from_address, date, body, file_name)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                email_data["message_id"],
                email_data["subject"],
                email_data["from"],
                date_str,
                email_data["body"],
                email_data["file_name"],
            ),
        )
        email_pk_id = cursor.lastrowid

        recipient_data = []
        recipient_data.extend([(email_pk_id, addr, "to") for addr in to_list])
        recipient_data.extend([(email_pk_id, addr, "cc") for addr in cc_list])
        recipient_data.extend([(email_pk_id, addr, "bcc") for addr in bcc_list])

        if recipient_data:
            cursor.executemany(
                """
                INSERT INTO recipients (email_id, recipient_address, recipient_type)
                VALUES (?, ?, ?)
                """,
                recipient_data,
            )
        record_count += 1

    conn.commit()
    conn.close()
    logging.info(f"Successfully loaded {record_count} email records.")


def create_indexes_and_triggers(db_path: str):
    """Create indexes and triggers on the populated database."""
    logging.info(f"Creating indexes and triggers for database: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    conn.commit()
    conn.close()
    logging.info("Successfully created indexes and triggers.")


def generate_database_environment(overwrite: bool = False):
    """
    Main function orchestrating the entire database environment creation process.

    Args:
        overwrite: If True, delete and recreate the database if it already exists.
    """
    logging.info(
        f"Starting database environment creation from repo '{REPO_ID}' at '{DB_PATH}'"
    )
    logging.info(f"Overwrite database if it exists: {overwrite}")

    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        logging.info(f"Creating data directory: {db_dir}")
        os.makedirs(db_dir)

    if os.path.exists(DB_PATH):
        if overwrite:
            logging.warning(f"Deleting existing database: {DB_PATH}")
            os.remove(DB_PATH)
        else:
            logging.warning(
                f"Database {DB_PATH} already exists and 'overwrite' is False. Skipping creation."
            )
            return

    dataset = download_dataset(REPO_ID)

    create_database(DB_PATH)

    populate_database(DB_PATH, dataset)

    create_indexes_and_triggers(DB_PATH)

    logging.info(f"Successfully created database environment at {DB_PATH}.")


if __name__ == "__main__":
    generate_database_environment(overwrite=True)