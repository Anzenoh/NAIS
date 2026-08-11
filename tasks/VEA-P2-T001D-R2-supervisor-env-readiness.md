# VEA-P2-T001D-R2 — Bind detached supervisor launch environment

## Objective

Make the reviewed persistent Apple runbook fail closed before detaching the
supervisor and prove that every variable referenced by either generated child
script is locally assigned, a shell special/positional parameter, or explicitly
exported by the parent.

## Preserved contracts

The T001D scientific plan, encoder and VLM identities, anchor transport,
selection, chunking, condition order, metrics, generation, storage accounting,
artifacts, atomic checkpoints, and resource-only resume behavior remain
unchanged.

## Local definition of done

- `RUN_TOKEN` and all remaining externally consumed bindings reach the detached
  supervisor.
- Every directory, checkout, resource, metadata, plan, generated-file, and
  syntax failure prevents `nohup` while leaving the parent tmux shell alive.
- Exact offline snapshot/root checks happen before the immutable plan and GPU
  waiting, with no runtime import or repair.
- Parsed here-document tests report zero unresolved external variables for both
  generated scripts.
