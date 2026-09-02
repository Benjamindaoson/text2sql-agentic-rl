import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.train_grpo import load_training_config, score_sql_reward


ROOT = Path(__file__).resolve().parents[1]


class GRPOLauncherConfigTests(unittest.TestCase):
    def test_v100_smoke_config_is_feasible(self):
        config = load_training_config(ROOT / "configs" / "gpu_grpo.json")
        self.assertEqual("fp16", config["precision"])
        self.assertLessEqual(config["model_parameter_billion"], 1.5)

    def test_v100_rejects_large_model(self):
        payload = json.loads((ROOT / "configs" / "gpu_grpo.json").read_text(encoding="utf-8"))
        payload["model_parameter_billion"] = 7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "single_v100"):
                load_training_config(path)

    def test_training_reward_uses_result_equivalence_and_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sample.sqlite"
            connection = sqlite3.connect(database)
            try:
                connection.execute("create table score (value integer)")
                connection.execute("insert into score values (7)")
                connection.commit()
            finally:
                connection.close()
            self.assertEqual(1.0, score_sql_reward(database, "select value from score", "select value from score"))
            self.assertEqual(-1.0, score_sql_reward(database, "delete from score", "select value from score"))
