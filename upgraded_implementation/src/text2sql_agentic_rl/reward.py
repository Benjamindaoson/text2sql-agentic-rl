from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Outcome(StrEnum):
    EQUIVALENT = "equivalent"
    EXECUTABLE_WRONG = "executable_wrong"
    EXECUTION_FAILURE = "execution_failure"
    PARSE_FAILURE = "parse_failure"
    SAFETY_REJECTED = "safety_rejected"


@dataclass(frozen=True)
class RewardBreakdown:
    correctness: float
    retry_penalty: float
    total: float


_CORRECTNESS = {
    Outcome.EQUIVALENT: 1.0,
    Outcome.EXECUTABLE_WRONG: 0.0,
    Outcome.EXECUTION_FAILURE: -0.2,
    Outcome.PARSE_FAILURE: -0.3,
    Outcome.SAFETY_REJECTED: -1.0,
}


def score_trajectory(outcome: Outcome, retry_count: int) -> RewardBreakdown:
    if retry_count < 0:
        raise ValueError("retry_count must be non-negative")
    correctness = _CORRECTNESS[outcome]
    retry_penalty = -0.03 * retry_count
    return RewardBreakdown(correctness, retry_penalty, correctness + retry_penalty)
