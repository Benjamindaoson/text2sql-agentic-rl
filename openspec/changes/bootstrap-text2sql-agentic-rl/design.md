## Context

The course Spider agent is a valuable legacy implementation, but it permits a generic SQL tool, allows the language model to terminate itself, and returns only terminal binary reward. The new project must preserve that baseline and introduce a separate safe environment and objective evaluator.

## Goals / Non-Goals

**Goals:** archive the original code, execute read-only SQLite safely, compare predicted/gold result semantics, record replayable traces, and provide a GPU GRPO launcher.

**Non-Goals:** edit the legacy source, train on CPU, expose Gold SQL to the agent, or claim financial-domain behavior.

## Decisions

- Use SQLite read-only URI plus Authorizer, not only SQL string matching, because a copied database alone does not enforce a safe tool boundary.
- Use a lexical allowlist before SQLite execution and reject multi-statement and mutation keywords.
- Treat `ORDER BY` in Gold SQL as order-sensitive; otherwise compare normalized row multisets so equivalent SQL syntax is accepted.
- Separate deterministic environment termination from optional LLM diagnosis; model text cannot certify correctness.
- Preserve the legacy Agent Lightning path and expose a modern GPU configuration separately, avoiding dependency mixing.

## Risks / Trade-offs

- [SQL parser is intentionally lightweight] → Authorizer and read-only URI remain the enforcement layer; lexical validation provides readable error types.
- [Timeout is cooperative in SQLite] → use a progress handler and record timeout state.
- [Spider is a benchmark, not finance] → retain task-specific naming and document the limitation.

## Migration Plan

Copy source read-only, verify CPU tests against temporary databases, then mount canonical Spider data on a GPU machine and run the dry-run before actual GRPO. Deleting the new project never affects the course directory.

## Open Questions

The exact modern Agent Lightning/veRL version is selected in GPU configuration after validating the target CUDA image.
