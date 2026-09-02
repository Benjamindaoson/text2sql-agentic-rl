from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .equivalence import compare_query_results
from .reward import Outcome, RewardBreakdown, score_trajectory
from .sandbox import SQLSafetyError
from .trace import AttemptTrace


class TraceState(StrEnum):
    SUCCESS = "SUCCESS"
    EXHAUSTED = "EXHAUSTED"
    SAFETY_REJECTED = "SAFETY_REJECTED"


@dataclass(frozen=True)
class AgentTrace:
    question: str
    state: TraceState
    attempts: list[AttemptTrace]
    final_reward: RewardBreakdown


class SQLAgentController:
    def __init__(self, database: Path, gold_sql: str, max_attempts: int = 3) -> None:
        self.database, self.gold_sql, self.max_attempts = database, gold_sql, max_attempts

    def run(self, question: str, attempts: list[str]) -> AgentTrace:
        traces: list[AttemptTrace] = []
        for index, sql in enumerate(attempts[: self.max_attempts]):
            result = compare_query_results(self.database, sql, self.gold_sql)
            if result.error:
                outcome = Outcome.SAFETY_REJECTED if result.error.startswith("SQLSafetyError") else Outcome.EXECUTION_FAILURE
            else:
                outcome = Outcome.EQUIVALENT if result.equivalent else Outcome.EXECUTABLE_WRONG
            reward = score_trajectory(outcome, index)
            traces.append(AttemptTrace(sql, outcome.value, reward.total, result.error))
            if outcome is Outcome.EQUIVALENT:
                return AgentTrace(question, TraceState.SUCCESS, traces, reward)
            if outcome is Outcome.SAFETY_REJECTED:
                return AgentTrace(question, TraceState.SAFETY_REJECTED, traces, reward)
        final = score_trajectory(Outcome.EXECUTION_FAILURE, max(len(traces) - 1, 0))
        return AgentTrace(question, TraceState.EXHAUSTED, traces, final)
