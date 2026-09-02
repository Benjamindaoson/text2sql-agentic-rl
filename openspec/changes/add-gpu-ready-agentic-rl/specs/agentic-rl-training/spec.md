# Agentic RL GPU training requirements

## ADDED Requirements

### Requirement: Spider tasks are runnable and auditable

系统 SHALL 生成包含数据库路径、问题和 gold SQL 的训练任务，并拒绝找不到 SQLite 数据库的样本。

#### Scenario: Missing database is rejected

- **WHEN** Spider `db_id` 没有对应数据库文件
- **THEN** 样本不进入 task JSONL 且审计记录 `missing_database`。

### Requirement: GPU profile declares feasibility

训练脚本 SHALL 仅接受已知 profile，V100 profile 必须 FP16 且模型规模不超过其配置上限。

#### Scenario: Infeasible full model is rejected on single V100

- **WHEN** `single_v100_smoke` 选择 7B 模型
- **THEN** dry-run 以资源契约错误失败。

### Requirement: Reward is environment-owned

rollout 的最终 reward SHALL 从 SQL 安全、执行结果等价和重试次数计算，而不是 SQL 字符串匹配。

#### Scenario: Equivalent non-identical SQL gets positive reward

- **WHEN** 生成 SQL 与 gold SQL 结果等价但文本不同
- **THEN** 轨迹获得等价奖励。
