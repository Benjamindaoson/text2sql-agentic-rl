import sys
import tempfile
import unittest
from pathlib import Path
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upgraded_implementation" / "src"))

from text2sql_agentic_rl.equivalence import compare_query_results
from text2sql_agentic_rl.reward import Outcome, score_trajectory


class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "sample.sqlite"
        conn = sqlite3.connect(self.db)
        try:
            conn.execute("CREATE TABLE numbers (n INTEGER)")
            conn.executemany("INSERT INTO numbers VALUES (?)", [(2,), (1,)])
            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        self.temp.cleanup()

    def test_unordered_rows_are_equivalent_and_ordered_rows_are_not(self):
        unordered = compare_query_results(self.db, "SELECT n FROM numbers ORDER BY n DESC", "SELECT n FROM numbers")
        ordered = compare_query_results(self.db, "SELECT n FROM numbers ORDER BY n", "SELECT n FROM numbers ORDER BY n DESC")
        self.assertTrue(unordered.equivalent)
        self.assertFalse(ordered.equivalent)

    def test_reward_distinguishes_safety_and_execution_failure(self):
        safety = score_trajectory(Outcome.SAFETY_REJECTED, retry_count=0)
        execution = score_trajectory(Outcome.EXECUTION_FAILURE, retry_count=0)
        self.assertLess(safety.total, execution.total)


if __name__ == "__main__":
    unittest.main()
