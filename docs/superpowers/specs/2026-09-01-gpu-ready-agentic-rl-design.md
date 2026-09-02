# 可验证 Text-to-SQL Agentic RL GPU-Ready 设计

## 目标

将课程 Agent Lightning/Spider 快照与已完成的只读 SQL 环境、结果等价奖励整合为可执行的训练链路，并区分单卡 smoke 与正式 veRL/Agent Lightning 运行。

## 架构与边界

`legacy_reproduction/` 保留课程版 Agent Lightning 代码；`text2sql_agentic_rl/` 提供唯一的环境真相来源：SQL 安全验证、SQLite 执行、gold/generation 结果等价及 reward。训练数据适配器生成包含 question、database、gold_sql 的 rollout task JSONL。

GPU 入口提供两个明确 profile：`single_v100_smoke` 仅针对 0.5B/1.5B 模型、FP16、小数据和短 rollout；`distributed_agentlightning_verl` 是正式 Agent Lightning + veRL 运行，要求多 GPU 或足够显存。脚本不得把单张 V100 32GB 宣称为 7B 正式 GRPO 环境。

## 验收标准

- CPU 可从 Spider JSON/SQLite 元数据构造 task JSONL，缺失数据库或危险 SQL 必须被审计。
- CPU dry-run 验证 profile、奖励权重、任务数据和模型配置，且不导入 GPU 依赖。
- 单卡训练入口实际建立当前 Agent Lightning/veRL 配置、启动 runner/trainer；正式 profile 在资源不足时 fail-fast。
- 训练后命令对固定 Spider 测试集同时报告 SQL 可执行率、结果等价率、Exact Match、平均重试和无效工具调用率。

## 非目标

CPU 不启动 vLLM、Ray、Agent Lightning server 或 Spider 模型 rollout。
