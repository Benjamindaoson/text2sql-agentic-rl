import sys
import tempfile
import unittest
from pathlib import Path
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upgraded_implementation" / "src"))

from text2sql_agentic_rl.controller import SQLAgentController, TraceState


class ControllerTests(unittest.TestCase):
    def test_controller_stops_on_equivalent_answer(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "sample.sqlite"
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE items (id INTEGER)")
                conn.execute("INSERT INTO items VALUES (1)")
                conn.commit()
            finally:
                conn.close()
            trace = SQLAgentController(db, "SELECT id FROM items", max_attempts=3).run(
                "show id", ["SELECT id FROM items", "DROP TABLE items"]
            )
        self.assertEqual(TraceState.SUCCESS, trace.state)
        self.assertEqual(1, len(trace.attempts))


if __name__ == "__main__":
    unittest.main()
