# 基于 Agentic RL 的 Text-to-SQL 决策智能体

项目保留课程的 Agent Lightning/Spider 代码快照，并把奖励基础升级为：只读 SQL 沙箱、执行结果等价、可解释轨迹和稠密奖励。结果等价而非 SQL 文本匹配是成功判定主标准。

CPU：`python scripts/prepare_course_materials.py`，然后 `python -m unittest discover -s tests -v`。

## GPU 训练（单卡 V100 32GB smoke）

将课程 `微调SQL数据集/data/` 整个目录上传或挂载到 GPU 主机；它必须同时包含 `train_spider.parquet`、`test_dev.parquet`、`database/` 与 `test_database/`。训练器直接复用 `legacy_reproduction/spider/train_sql_agent.py` 的 Agent Lightning/veRL 训练逻辑，同时把课程的 10 runners、4 个生成和大 batch 收敛为单卡 smoke 参数。

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-gpu.txt
python scripts/train_grpo.py --dry-run
python scripts/train_grpo.py --spider-data-root /mnt/data/spider/data
```

GPU 启动前会验证 FP16、模型大小、两份 Spider parquet、数据库目录与 CUDA；启动器在运行时将课程 Agent 的 `evaluate_query` 替换为升级实现中的只读沙箱与执行结果等价 reward，课程原文件不被修改。
