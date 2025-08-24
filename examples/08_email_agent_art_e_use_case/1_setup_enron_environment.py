# ============================================================
# NOTE: Before running this script, please create a folder:
#        TOOLBRAIN/data
#
# This directory is required to store datasets and outputs.
#
# 1. Environment Setup (run only once):
# This script will automatically download the Enron dataset 
# and create the file enron_emails.db
# python -m examples.08_email_agent_art_e_use_case.1_setup_enron_environment
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

-- Tạo bảng ảo FTS5 để tìm kiếm toàn văn trên subject và body
CREATE VIRTUAL TABLE emails_fts USING fts5(
    subject,
    body,
    content='emails',
    content_rowid='id'
);

-- Các trigger để tự động đồng bộ dữ liệu giữa bảng 'emails' và 'emails_fts'
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

-- Nạp dữ liệu ban đầu vào bảng FTS
INSERT INTO emails_fts (rowid, subject, body) SELECT id, subject, body FROM emails;
"""




def download_dataset(repo_id: str) -> Dataset:
    """Tải dataset từ Hugging Face Hub."""
    logging.info(f"Đang tải dataset từ Hugging Face Hub: {repo_id}")
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
        raise TypeError(f"Lỗi: Kiểu dữ liệu mong đợi là Dataset, nhận được {type(dataset_obj)}")
    logging.info(
        f"Tải thành công dataset '{repo_id}' với {len(dataset_obj)} bản ghi."
    )
    return dataset_obj


def create_database(db_path: str):
    """Tạo database SQLite và các bảng."""
    logging.info(f"Đang tạo database SQLite và các bảng tại: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()
    conn.close()
    logging.info("Tạo các bảng trong database thành công.")


def populate_database(db_path: str, dataset: Dataset):
    """Nạp dữ liệu từ dataset Hugging Face vào database."""
    logging.info(f"Đang nạp dữ liệu vào database {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")

    record_count = 0
    
    conn.execute("BEGIN TRANSACTION;")

    for email_data in tqdm(dataset, desc="Đang ghi emails vào DB"):
        assert isinstance(email_data, dict)
        
        # Trích xuất và làm sạch dữ liệu
        date_obj: datetime = email_data["date"]
        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S") if date_obj else None
        
        to_list = [str(addr) for addr in email_data.get("to", []) if addr]
        cc_list = [str(addr) for addr in email_data.get("cc", []) if addr]
        bcc_list = [str(addr) for addr in email_data.get("bcc", []) if addr]

        # Ghi vào bảng 'emails'
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
    logging.info(f"Nạp thành công {record_count} bản ghi email.")


def create_indexes_and_triggers(db_path: str):
    """Tạo các chỉ mục và trigger trên database đã có dữ liệu."""
    logging.info(f"Đang tạo indexes và triggers cho database: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    conn.commit()
    conn.close()
    logging.info("Tạo indexes và triggers thành công.")


def generate_database_environment(overwrite: bool = False):
    """
    Hàm chính điều phối toàn bộ quá trình tạo môi trường database.

    Args:
        overwrite: Nếu True, xóa và tạo lại database nếu đã tồn tại.
    """
    logging.info(
        f"Bắt đầu quá trình tạo môi trường database từ repo '{REPO_ID}' tại '{DB_PATH}'"
    )
    logging.info(f"Ghi đè database nếu tồn tại: {overwrite}")

    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        logging.info(f"Đang tạo thư mục data: {db_dir}")
        os.makedirs(db_dir)

    if os.path.exists(DB_PATH):
        if overwrite:
            logging.warning(f"Đang xóa database đã tồn tại: {DB_PATH}")
            os.remove(DB_PATH)
        else:
            logging.warning(
                f"Database {DB_PATH} đã tồn tại và 'overwrite' là False. Bỏ qua quá trình tạo."
            )
            return

    dataset = download_dataset(REPO_ID)

    create_database(DB_PATH)

    populate_database(DB_PATH, dataset)

    create_indexes_and_triggers(DB_PATH)

    logging.info(f"Hoàn tất quá trình tạo môi trường database tại {DB_PATH}.")


if __name__ == "__main__":
    generate_database_environment(overwrite=True)