import sys
import tempfile
import unittest
from pathlib import Path
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upgraded_implementation" / "src"))

from text2sql_agentic_rl.sandbox import SQLSafetyError, execute_readonly, validate_read_only_sql


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "sample.sqlite"
        conn = sqlite3.connect(self.db)
        try:
            conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
            conn.executemany("INSERT INTO items VALUES (?, ?)", [(1, "a"), (2, "b"), (3, "c")])
            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        self.temp.cleanup()

    def test_safe_select_and_cte(self):
        validate_read_only_sql("SELECT name FROM items")
        validate_read_only_sql("WITH x AS (SELECT * FROM items) SELECT name FROM x")
        result = execute_readonly(self.db, "SELECT name FROM items", row_limit=2, timeout_ms=1_000)
        self.assertEqual([("a",), ("b",)], result.rows)
        self.assertTrue(result.truncated)

    def test_mutation_and_multiple_statements_are_rejected(self):
        for sql in ("DROP TABLE items", "SELECT * FROM items; DELETE FROM items"):
            with self.assertRaises(SQLSafetyError):
                validate_read_only_sql(sql)


if __name__ == "__main__":
    unittest.main()
