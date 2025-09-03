"""
Email Environment Tools for the ART-E Use Case.

This module provides the necessary tools for the ToolBrain agent to interact
with the Enron email database (enron_emails.db). The agent will use these
tools to search for and read emails to answer user queries.

The logic is a direct adaptation of the original tools from the ART-E project
to ensure a consistent and effective environment for the agent.
"""

import sqlite3
import logging
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict

from smolagents import tool

# from . import config
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))
import config


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "..", "data", "enron_emails.db")

_db_connection = None


def _get_db_connection() -> sqlite3.Connection:
    """
    Establishes and returns a thread-safe, read-only SQLite connection.
    This function ensures that the database is only connected to once per run.
    """
    global _db_connection
    if _db_connection is None:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(
                f"Database file not found at {DB_PATH}. "
                "Please run '1_setup_enron_environment.py' first."
            )
        # Connect in read-only mode ('ro') for safety.
        # check_same_thread=False is necessary for some environments.
        _db_connection = sqlite3.connect(
            f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False
        )
    return _db_connection


# --- Data Structures for Tool Outputs ---


@dataclass
class SearchResult:
    """
    A dataclass to structure the results of an email search.
    Provides a clean, predictable output for the agent.
    """

    message_id: str
    snippet: str


@dataclass
class EmailDetails:
    """
    A dataclass to hold the full details of a retrieved email.
    """

    message_id: str
    date: str
    subject: str
    from_address: str
    to_addresses: List[str]
    cc_addresses: List[str]
    bcc_addresses: List[str]
    body: str


# --- Agent Tools ---


@tool
def search_emails(
    keywords: List[str],
    inbox: Optional[str] = None,
    from_addr: Optional[str] = None,
    to_addr: Optional[str] = None,
    sent_after: Optional[str] = None,
    sent_before: Optional[str] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Searches the email database based on keywords, sender, recipient, and date range.

    Args:
        keywords: A list of keywords that must all appear in the subject or body.
        inbox: Optional. The user's email address. If provided, results will include
               emails sent from OR to (inc. cc/bcc) this address.
        from_addr: Optional. An email address to filter emails sent *from*.
        to_addr: Optional. An email address to filter emails sent *to* (inc. cc/bcc).
        sent_after: Optional. A date string 'YYYY-MM-DD'. Filters for emails on or after this date.
        sent_before: Optional. A date string 'YYYY-MM-DD'. Filters for emails before this date.
        max_results: The maximum number of results to return. Cannot exceed 10.

    Returns:
        A list of dictionaries, each containing 'message_id' and 'snippet'.
        Returns an empty list if no results are found.
    """
    if not keywords:
        return [{"error": "No keywords provided for search."}]
    if max_results > 10:
        logging.warning("max_results exceeds 10. Capping at 10.")
        max_results = 10

    conn = _get_db_connection()
    cursor = conn.cursor()

    where_clauses: List[str] = []
    params: List[Any] = []

    # 1. Keywords (FTS5 search)
    # FTS5's default operator is AND. We escape quotes for safety.
    # fts_query = " ".join(f'"{k.replace("\"", "\"\"")}"' for k in keywords)
    keyword_terms = [k.replace('"', '""') for k in keywords]
    fts_query = " ".join(f'"{term}"' for term in keyword_terms)
    where_clauses.append("emails_fts MATCH ?")
    params.append(fts_query)

    # 2. Inbox filter (must be from OR to/cc/bcc the user)
    if inbox:
        where_clauses.append(
            """(e.from_address = ? OR EXISTS (
                SELECT 1 FROM recipients r_inbox
                WHERE r_inbox.recipient_address = ? AND r_inbox.email_id = e.id
            ))"""
        )
        params.extend([inbox, inbox])

    # 3. Optional From filter
    if from_addr:
        where_clauses.append("e.from_address = ?")
        params.append(from_addr)

    # 4. Optional To filter (includes to, cc, bcc)
    if to_addr:
        where_clauses.append(
            """EXISTS (
                SELECT 1 FROM recipients r_to
                WHERE r_to.recipient_address = ? AND r_to.email_id = e.id
            )"""
        )
        params.append(to_addr)

    # 5. Optional Date filters
    if sent_after:
        where_clauses.append("e.date >= ?")
        params.append(f"{sent_after} 00:00:00")
    if sent_before:
        where_clauses.append("e.date < ?")
        params.append(f"{sent_before} 00:00:00")

    # Construct the final SQL query
    sql = f"""
        SELECT
            e.message_id,
            snippet(emails_fts, -1, '<b>', '</b>', ' ... ', 15) as snippet
        FROM
            emails e JOIN emails_fts fts ON e.id = fts.rowid
        WHERE
            {' AND '.join(where_clauses)}
        ORDER BY
            e.date DESC
        LIMIT ?;
    """
    params.append(max_results)

    try:
        cursor.execute(sql, params)
        results = cursor.fetchall()
        formatted_results = [
            asdict(SearchResult(message_id=row[0], snippet=row[1])) for row in results
        ]
        logging.info(f"Search for '{keywords}' found {len(formatted_results)} results.")
        return formatted_results
    except sqlite3.Error as e:
        logging.error(f"Database search error: {e}")
        return [{"error": f"A database error occurred: {e}"}]


@tool
def read_email(message_id: str) -> Dict[str, Any]:
    """
    Retrieves the full details of a single email by its message_id.

    Args:
        message_id: The unique identifier of the email to retrieve (e.g., '<1234.JavaMail.evans@thyme>').

    Returns:
        A dictionary containing the email's details, or an error dictionary if not found.
    """
    conn = _get_db_connection()
    cursor = conn.cursor()

    try:
        # Query for the main email content
        cursor.execute(
            "SELECT date, subject, from_address, body FROM emails WHERE message_id = ?;",
            (message_id,),
        )
        email_row = cursor.fetchone()

        if not email_row:
            logging.warning(f"Email with message_id '{message_id}' not found.")
            return {"error": f"Email with message_id '{message_id}' not found."}

        date, subject, from_addr, body = email_row

        # Check if the limit is set in the config and if the body exceeds it.
        if config.MAX_TOOL_OUTPUT_CHARS and len(body) > config.MAX_TOOL_OUTPUT_CHARS:
            logging.warning(
                f"Truncating long email body for message_id '{message_id}'. "
                f"Original length: {len(body)}, Limit: {config.MAX_TOOL_OUTPUT_CHARS}."
            )
            # Truncate the string and add a clear indicator for the agent.
            body = body[:config.MAX_TOOL_OUTPUT_CHARS] + "\n... [Content truncated due to length limit]"

        # Query for all recipients (to, cc, bcc)
        cursor.execute(
            "SELECT recipient_address, recipient_type FROM recipients WHERE email_id = (SELECT id FROM emails WHERE message_id = ?);",
            (message_id,),
        )
        recipient_rows = cursor.fetchall()

        to_addrs, cc_addrs, bcc_addrs = [], [], []
        for addr, type in recipient_rows:
            if type == "to":
                to_addrs.append(addr)
            elif type == "cc":
                cc_addrs.append(addr)
            elif type == "bcc":
                bcc_addrs.append(addr)

        email_details = EmailDetails(
            message_id=message_id,
            date=date,
            subject=subject,
            from_address=from_addr,
            to_addresses=to_addrs,
            cc_addresses=cc_addrs,
            bcc_addresses=bcc_addrs,
            body=body,
        )

        logging.info(f"Successfully read email with message_id '{message_id}'.")
        return asdict(email_details)

    except sqlite3.Error as e:
        logging.error(f"Database read error for message_id '{message_id}': {e}")
        return {"error": f"A database error occurred while reading the email: {e}"}
