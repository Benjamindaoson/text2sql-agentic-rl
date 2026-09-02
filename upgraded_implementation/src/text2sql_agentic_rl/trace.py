from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any


@dataclass(frozen=True)
class AttemptTrace:
    sql: str
    outcome: str
    reward: float
    error: str | None


def to_json(value: Any) -> str:
    return json.dumps(asdict(value), ensure_ascii=False, indent=2)
