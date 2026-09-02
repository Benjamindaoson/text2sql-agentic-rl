# GPU-ready Text-to-SQL Agentic RL trainer

## Why

当前项目的环境与奖励可验证，但训练入口只做依赖检查。

## What Changes

- 新增 Spider task 数据适配与质量审计。
- 新增受硬件 profile 约束的 Agent Lightning/veRL 训练启动器。
- 新增训练前后评测命令与 CPU dry-run 测试。

## Impact

影响 Text-to-SQL 项目的升级实现、配置、脚本、测试和 README；不改课程快照。
