## ADDED Requirements

### Requirement: Immutable Spider course snapshot
The project SHALL copy the Spider source files without editing their contents and record source file hashes and the canonical Spider dataset path.

#### Scenario: Source integrity
- **WHEN** material preparation is run
- **THEN** the manifest records the source hash and the copied legacy source hash is identical
