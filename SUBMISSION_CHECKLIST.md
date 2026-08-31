# Kopibara submission checklist

## Local requirements

| Requirement | Status | Evidence |
|---|---|---|
| Reproducible setup and installation | PASS | `pyproject.toml`, `uv.lock`, `Makefile` |
| Well-structured commented public code | PASS | `src/kopibara_agent/`, `experiments/`, `scripts/` |
| README overview and reproduction steps | PASS | `README.md` |
| Tools, API, libraries, data, and assets described | PASS | `SUBMISSION_DESCRIPTION.md` |
| Limitations and future work described | PASS | `SUBMISSION_DESCRIPTION.md` |
| Official starter kit verified | PASS | `make verify-starter` |
| Required KuaiRand-Pure run | PASS | `runs/champion/20260831T142500Z/` |
| Per-iteration hypotheses, diffs, metrics, recovery events | PASS | `runs/champion/20260831T142500Z/` and autonomous logs |
| Manual-intervention count reported | PASS | latest autonomous manifest |
| Resource usage reported | PASS | latest autonomous manifest |
| Final output in organizer schema | PASS | ensemble `submission.csv` and untouched `submit.py --check` |
| Validation-best result and baseline delta reported | PASS | `SUBMISSION_DESCRIPTION.md` |
| Video | NOT REQUIRED | Organizer marks it recommended, not required |
| Optional KuaiRand-1K execution | PASS | `runs/lightgbm-binary-1k/seed0/` and generic checker |

## Deliberately out of scope for this local verification

| Requirement | Status | Action |
|---|---|---|
| Public GitHub repository | NOT CHECKED | User requested local result only |
| Devpost written description | NOT CHECKED | User requested local result only |
| Final hidden-test score | NOT AVAILABLE LOCALLY | The organizer scores hidden test once |

All locally verifiable requirements pass. The workspace has no configured Git
remote; publication and hidden-test scoring are intentionally not part of this
local check.
