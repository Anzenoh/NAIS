# <TASK_ID> — <Task title>

## Mode

`investigation-only` | `implementation` | `validation-only` | `documentation`

## Goal

State one externally verifiable outcome.

## Why this task exists

Explain the dependency or research question this task addresses.

## Read first

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `<task-specific files>`

## Current evidence

List existing files, errors, outputs, or known environment facts. Use `unknown` rather than guessing.

## Scope

### Allowed changes

- exact files or directories;
- exact behavior permitted.

### Non-goals

- explicit work that must not be done;
- future tasks that must not be pulled forward.

## Constraints

- research invariants;
- compatibility or dependency limits;
- network and access policy;
- data and secret policy;
- backward-compatibility requirements.

## Deliverables

1. Exact file or behavior.
2. Tests or validation tooling.
3. Documentation or manifest updates.
4. Run report path.

## Validation

### Required

- command or observable check;
- expected exit status or invariant.

### Optional

- useful checks that do not determine PASS.

## Stop conditions

Stop and report `BLOCKED` rather than improvising when:

- a required input or license is unavailable;
- the requested change conflicts with a protected contract;
- a dependency or model substitution would be required;
- unrelated user changes would be overwritten;
- a required check cannot be made meaningful.

## Required final response

Use the format in `AGENTS.md` and create a report from `docs/templates/RUN_REPORT_TEMPLATE.md`.
