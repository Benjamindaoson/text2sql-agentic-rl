# Text-to-SQL Agentic RL GPU-Ready Implementation Plan

> **For agentic workers:** Execute task-by-task with one CPU test cycle per task.

**Goal:** Turn the course Agent Lightning/veRL guard into a profile-validated launcher backed by auditable Spider tasks.

**Architecture:** The existing SQL environment stays the reward authority. A stdlib task adapter produces rows that reference SQLite databases; a GPU launcher calls the preserved Agent Lightning agent only after profile/data/GPU checks.

### Task 1: Spider task preparation

- [ ] Add a task builder and test rejecting missing SQLite databases.
- [ ] Run test red, implement, run green, commit.

### Task 2: Profiled Agent Lightning launcher

- [ ] Add config/dry-run tests for single-V100 FP16 small-model and distributed profile constraints.
- [ ] Replace `train_grpo.py` guard with an Agent Lightning/veRL launcher that forwards training and validation tasks.
- [ ] Run full CPU tests, compile, dry-run, commit.
