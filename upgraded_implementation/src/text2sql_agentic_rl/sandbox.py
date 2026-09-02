from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import time


class SQLSafetyError(ValueError):
    pass


class SQLTimeoutError(TimeoutError):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    rows: list[tuple]
    columns: tuple[str, ...]
    truncated: bool


_FORBIDDEN = re.compile(r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|VACUUM|ATTACH|DETACH|PRAGMA|REINDEX|ANALYZE)\b", re.I)


def _without_literals_and_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", "?", sql)


def validate_read_only_sql(sql: str) -> None:
    stripped = sql.strip()
    if not stripped:
        raise SQLSafetyError("empty_sql")
    statements = [part.strip() for part in stripped.split(";") if part.strip()]
    if len(statements) != 1:
        raise SQLSafetyError("multiple_statements")
    normalized = _without_literals_and_comments(statements[0])
    if not re.match(r"^(?:SELECT\b|WITH\b)", normalized, flags=re.I):
        raise SQLSafetyError("only_select_or_with_select_allowed")
    if _FORBIDDEN.search(normalized):
        raise SQLSafetyError("forbidden_sql_operation")
    if normalized.upper().startswith("WITH") and not re.search(r"\bSELECT\b", normalized, flags=re.I):
        raise SQLSafetyError("cte_must_contain_select")


def _authorizer(action: int, arg1: str | None, arg2: str | None, database: str | None, source: str | None) -> int:
    denied = {
        sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_ALTER_TABLE, sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TRIGGER, sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_PRAGMA, sqlite3.SQLITE_TRANSACTION,
    }
    return sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK


def execute_readonly(database: Path, sql: str, row_limit: int = 200, timeout_ms: int = 5_000) -> ExecutionResult:
    validate_read_only_sql(sql)
    if row_limit < 1 or timeout_ms < 1:
        raise ValueError("row_limit and timeout_ms must be positive")
    uri = database.resolve().as_uri() + "?mode=ro"
    started = time.monotonic()
    rows: list[tuple]
    columns: tuple[str, ...]
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.set_authorizer(_authorizer)
        def progress() -> int:
            return 1 if (time.monotonic() - started) * 1_000 > timeout_ms else 0
        conn.set_progress_handler(progress, 1_000)
        try:
            cursor = conn.execute(sql)
            try:
                rows = cursor.fetchmany(row_limit + 1)
                columns = tuple(d[0] for d in cursor.description or ())
            finally:
                cursor.close()
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise SQLTimeoutError("sql_timeout") from exc
            raise
        finally:
            conn.set_progress_handler(None, 0)
    finally:
        conn.close()
    return ExecutionResult(rows=rows[:row_limit], columns=columns, truncated=len(rows) > row_limit)
