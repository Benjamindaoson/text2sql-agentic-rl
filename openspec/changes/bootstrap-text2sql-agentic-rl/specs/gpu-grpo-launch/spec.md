## ADDED Requirements

### Requirement: GPU GRPO dry run
The GRPO launcher SHALL validate its configuration and report required GPU stack versions before training begins.

#### Scenario: Missing CUDA
- **WHEN** the launcher is called without CUDA and without `--dry-run`
- **THEN** it exits with an actionable GPU requirement message before creating training artifacts
