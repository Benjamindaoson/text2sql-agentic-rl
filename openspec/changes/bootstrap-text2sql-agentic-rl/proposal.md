## Why

The course Spider example demonstrates Agent Lightning and GRPO but relies on an old interface, a permissive SQL tool, model-controlled self-check termination, and terminal binary reward. This change preserves the course implementation while creating a safe, verifiable, CPU-tested upgraded path.

## What Changes

- Copy the course Spider source unchanged into a legacy reproduction directory and record source integrity.
- Add a read-only SQLite sandbox, result-equivalence evaluator, reward decomposition, deterministic controller, and JSON traces.
- Add GPU configuration and a dry-runnable GRPO launch entrypoint without starting training locally.

## Capabilities

### New Capabilities
- `course-provenance`: Preserves course Spider source and canonical-data references.
- `safe-sql-environment`: Executes only bounded read-only SQL in SQLite.
- `sql-result-equivalence`: Evaluates predicted and gold SQL by execution result semantics.
- `agent-trajectory-reward`: Produces deterministic traces and decomposed rewards.
- `gpu-grpo-launch`: Validates the legacy/modern GPU training configuration.

### Modified Capabilities

- None.

## Impact

Adds a standalone Python project and does not alter the course Spider source or duplicate the full Spider database directory.
