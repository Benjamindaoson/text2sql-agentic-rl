## ADDED Requirements

### Requirement: Deterministic trajectory rewards
The reward module SHALL distinguish equivalent success, executable wrong answer, execution failure, parse failure, safety rejection, and retry cost.

#### Scenario: Safety rejection
- **WHEN** an attempt is rejected by the SQL safety layer
- **THEN** its reward includes a negative safety component larger in magnitude than an ordinary execution failure

### Requirement: Environment-controlled termination
The controller SHALL terminate on equivalent success, safety rejection, or budget exhaustion rather than on a language-model self-certification string.

#### Scenario: Correct environmental result
- **WHEN** an attempt is result-equivalent
- **THEN** the trace state is `SUCCESS` and no additional attempt is executed
