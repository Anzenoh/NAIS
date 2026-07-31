# VEA-P1-T003B-A-R1 — SBR1 acceptance provenance gate repair

## Status

`PASS` pending the authorized commit and push recorded by Git delivery.

## Baseline and branch

- Baseline branch and HEAD: `main` / `e518a51ba83f70b42611cc4386a9f55dc9e97de3`
- Task branch: `codex/vea-p1-t003b-a-r1`
- SBR1 Run Report was read-only and remains SHA-256 `8abfc27089b5a91b20a9c6a37dd042a7b0375914dbafd76b6433267155c8cbcd`.

## Change

The T003B runbook no longer treats the historical SBR1 Run Report as the operational
acceptance-marker source. After successfully verifying the user-supplied R1R3 Task
SHA-256, it fail-closes unless that verified R1R3 Task contains exactly
`VEA-P1-T002D-SBR1 = ACCEPTED_PASS`. The check remains before evidence-root creation.

## Identity and changed paths

- Runbook before SHA-256: `416c3270ac222b6d8ef692957c2fb91562f5e18c95418083663e4a1b63abf36a`
- Runbook after SHA-256: `c254e815df03509e7a97c63fb3881087e18722ea5d55e6f3fdd31e26337a5896`
- Changed paths: this Task, this Run Report, the T003B runbook, and
  `tests/integration/test_p1_feature_extraction_cli.py` only.

## Validation

```text
py -3 -B -m pytest -q -p no:cacheprovider tests/integration/test_p1_feature_extraction_cli.py
exit = 0
23 passed

git -c core.whitespace=cr-at-eol diff --check
exit = 0
```

The test proves that the marker is read from `R1R3_TASK`, the R1R3 Task SHA check
precedes that marker check, the marker check precedes `mkdir "$EVIDENCE_ROOT"`, and
the Stage A/B plus exact `00` through `05` artifact boundaries remain present.

## Boundaries

```text
Stage A executed = false
Stage B executed = false
smoke/full extraction executed = false
package created = false
Apple/dataset/checkpoint/model/GPU access = false
```
