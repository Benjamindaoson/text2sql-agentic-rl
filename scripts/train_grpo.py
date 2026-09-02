"""Launch the course Spider Agent Lightning/veRL GRPO trainer on one GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "run_name", "hardware_profile", "base_model", "model_parameter_billion",
    "spider_data_root", "output_dir", "precision", "max_prompt_length",
    "max_completion_length", "num_generations", "learning_rate",
    "per_device_train_batch_size", "gradient_accumulation_steps", "reward",
}


def load_training_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or REQUIRED - config.keys():
        raise ValueError("missing required Spider GRPO training config fields")
    if config["hardware_profile"] == "single_v100":
        if config["precision"] != "fp16" or float(config["model_parameter_billion"]) > 1.5:
            raise ValueError("single_v100 requires fp16 and a model_parameter_billion value no larger than 1.5")
    if int(config["num_generations"]) < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    return config


def score_sql_reward(database: Path, predicted_sql: str, gold_sql: str) -> float:
    """Course-compatible scalar reward backed by the upgraded safe equivalence environment."""
    sys.path.insert(0, str(ROOT / "upgraded_implementation" / "src"))
    from text2sql_agentic_rl.equivalence import compare_query_results
    from text2sql_agentic_rl.reward import Outcome, score_trajectory

    result = compare_query_results(database, predicted_sql, gold_sql)
    if result.error:
        outcome = Outcome.SAFETY_REJECTED if result.error.startswith("SQLSafetyError") else Outcome.EXECUTION_FAILURE
    else:
        outcome = Outcome.EQUIVALENT if result.equivalent else Outcome.EXECUTABLE_WRONG
    return score_trajectory(outcome, retry_count=0).total


def run_training(config: Mapping[str, object]) -> None:
    data_root = Path(str(config["spider_data_root"])).expanduser()
    train_file, validation_file = data_root / "train_spider.parquet", data_root / "test_dev.parquet"
    missing = [str(path) for path in (train_file, validation_file, data_root / "database", data_root / "test_database") if not path.exists()]
    if missing:
        raise FileNotFoundError("Spider parquet files required: " + ", ".join(missing))
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi is required for Spider GRPO training")
    subprocess.run(["nvidia-smi"], check=True)
    try:
        import torch
        import agentlightning as agl
    except ImportError as error:
        raise RuntimeError("install GPU dependencies: torch agentlightning[verl] ray vllm pandas") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable after nvidia-smi preflight")
    course_root = ROOT / "legacy_reproduction" / "spider"
    sys.path.insert(0, str(course_root))
    import train_sql_agent as course

    os.environ["VERL_SPIDER_DATA_DIR"] = str(data_root.resolve())

    def verifiable_course_reward(query: str, ground_truth: str, database: str, raise_on_error: bool = True) -> float:
        return score_sql_reward(Path(database), query, ground_truth)

    course.evaluate_query = verifiable_course_reward

    training = course.config_train_fast()
    training["data"].update({
        "train_files": str(train_file), "val_files": str(validation_file),
        "train_batch_size": int(config["per_device_train_batch_size"]),
        "max_prompt_length": int(config["max_prompt_length"]),
        "max_response_length": int(config["max_completion_length"]),
    })
    rollout = training["actor_rollout_ref"]["rollout"]
    rollout.update({"n": int(config["num_generations"]), "gpu_memory_utilization": 0.6})
    training["actor_rollout_ref"]["model"]["path"] = str(config["base_model"])
    training["actor_rollout_ref"]["actor"].update({"ppo_mini_batch_size": 1, "ppo_micro_batch_size_per_gpu": 1, "optim": {"lr": float(config["learning_rate"])}})
    training["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] = 1
    training["trainer"].update({"experiment_name": str(config["run_name"]), "logger": ["console"], "total_epochs": 1})
    os.environ.setdefault("WANDB_MODE", "disabled")
    agent = course.LitSQLAgent()
    trainer = agl.Trainer(n_runners=1, algorithm=agl.VERL(training), adapter={"agent_match": "sql_agent"})
    train_data = course.pd.read_parquet(train_file).to_dict(orient="records")
    validation_data = course.pd.read_parquet(validation_file).to_dict(orient="records")
    trainer.fit(agent, train_dataset=train_data, val_dataset=validation_data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the preserved Spider agent with verifiable GRPO rewards")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "gpu_grpo.json")
    parser.add_argument("--spider-data-root", type=Path, help="directory containing parquet, database, and test_database")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_training_config(args.config)
    if args.spider_data_root:
        config = {**config, "spider_data_root": str(args.spider_data_root)}
    if args.dry_run:
        print(json.dumps({"mode": "dry_run", "run_name": config["run_name"], "profile": config["hardware_profile"], "spider_data_root": config["spider_data_root"]}, ensure_ascii=False))
        return 0
    run_training(config)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
