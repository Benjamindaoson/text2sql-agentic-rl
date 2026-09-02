## ADDED Requirements

### Requirement: Execution-result equivalence
The evaluator SHALL execute predicted and Gold SQL in the same safe environment and score equivalence by result semantics rather than SQL text identity.

#### Scenario: Unordered equivalent rows
- **WHEN** Gold SQL has no `ORDER BY` and predicted SQL returns the same rows in another order
- **THEN** the evaluator reports equivalent results

### Requirement: Ordered-result equivalence
The evaluator SHALL compare row sequence when the Gold SQL contains `ORDER BY`.

#### Scenario: Ordered mismatch
- **WHEN** Gold SQL includes `ORDER BY` and predicted rows have a different sequence
- **THEN** the evaluator reports non-equivalence
