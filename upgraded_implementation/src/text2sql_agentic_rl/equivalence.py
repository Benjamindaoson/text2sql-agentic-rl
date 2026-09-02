from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

from .sandbox import ExecutionResult, execute_readonly


@dataclass(frozen=True)
class EquivalenceResult:
    equivalent: bool
    order_sensitive: bool
    predicted: ExecutionResult | None
    gold: ExecutionResult | None
    error: str | None = None


def _normalize_cell(value: object) -> object:
    return round(value, 9) if isinstance(value, float) else value


def _normalize_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_normalize_cell(cell) for cell in row) for row in rows]


def _has_order_by(sql: str) -> bool:
    stripped = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", "", sql)
    return bool(re.search(r"\bORDER\s+BY\b", stripped, flags=re.I))


def compare_query_results(database: Path, predicted_sql: str, gold_sql: str, row_limit: int = 200, timeout_ms: int = 5_000) -> EquivalenceResult:
    order_sensitive = _has_order_by(gold_sql)
    try:
        predicted = execute_readonly(database, predicted_sql, row_limit, timeout_ms)
        gold = execute_readonly(database, gold_sql, row_limit, timeout_ms)
    except Exception as exc:
        return EquivalenceResult(False, order_sensitive, None, None, f"{type(exc).__name__}: {exc}")
    predicted_rows, gold_rows = _normalize_rows(predicted.rows), _normalize_rows(gold.rows)
    equivalent = predicted_rows == gold_rows if order_sensitive else Counter(predicted_rows) == Counter(gold_rows)
    return EquivalenceResult(equivalent, order_sensitive, predicted, gold)
