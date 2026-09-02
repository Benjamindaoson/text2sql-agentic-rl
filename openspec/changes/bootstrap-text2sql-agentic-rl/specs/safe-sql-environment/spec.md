## ADDED Requirements

### Requirement: Bounded read-only SQL execution
The SQL environment SHALL accept one `SELECT` or `WITH ... SELECT` statement and SHALL reject mutation, attachment, pragma, and multi-statement SQL before returning rows.

#### Scenario: Mutation rejection
- **WHEN** an agent submits `DROP TABLE users`
- **THEN** the environment returns a safety-rejection result and does not execute the statement

### Requirement: Resource-bound execution
The SQL environment SHALL enforce a configurable result-row limit and cooperative execution timeout.

#### Scenario: Result cap
- **WHEN** a query produces more rows than the configured limit
- **THEN** the result is marked truncated and contains no more than that limit
