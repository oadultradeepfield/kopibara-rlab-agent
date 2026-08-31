# Kopibara — Autonomous Machine Learning Research Agent

## Overview

Kopibara is an autonomous ML research agent for the KuaiRand recommender
systems challenge. It treats model development as a code-search problem: the
agent proposes a measurable hypothesis, edits a runnable candidate, executes
it on train/validation data, records the result, and keeps the best solution.
The system is an agent product, not only a recommender model.

## How it addresses the challenge

The controller:

1. Runs and records the organizer-provided FM baseline.
2. Starts from the measured historical-feature LambdaRank branch because the
   scored metrics are GAUC and nDCG@5 while the official seed uses pointwise
   logloss and no history features.
3. Sends the current solution tree, metrics, code, and research guidance to
   `gpt-5.6-luna`.
4. Validates each structured plan, applies exact patches, syntax-checks the
   result, and runs it with a timeout.
5. Retries a failed candidate with an automatically generated repair patch.
6. Selects the validation-best node, writes the aligned submission, and runs
   the untouched starter-kit checker.

Generated candidates cannot alter the evaluator, controller, or submission
checker. They can patch only the runnable candidate experiment, while the
controller logs the hypothesis, diff, metrics, resource use, and recovery
events for every iteration.

## Results from the verified run

| KuaiRand-Pure validation | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Organizer FM reference | 0.6674 | 0.5357 | 0.6016 |
| Kopibara historical-ranker ensemble | 0.7007 | 0.5520 | 0.6264 |
| Absolute delta | +0.0333 | +0.0163 | +0.0248 |

The winning branch is a grouped LightGBM LambdaRank ranker. Its features are
constructed chronologically and include prior feedback counts/rates at four
scopes: user, video, author, and user-video. The history is updated only after
train and public-validation rows; hidden-test rows are never used to update it.
Three independent seeds are z-normalized and averaged for the final Pure
submission. The three measured seed primaries were 0.6257, 0.6263, and 0.6265;
the ensemble validation primary was 0.6264.

This result is validation-only. The organizer's hidden-test score is not
available locally, so no hidden score is claimed.

The original controlled run used 7 autonomous candidate iterations, 29,569 LLM
tokens, 265 seconds, and 0 manual interventions. The historical-ranker
ensemble itself used three local model seeds and no LLM tokens. The organizer
submission checker accepted all 170,588 rows. No hidden-test labels were used
in feature construction or model selection.

The optional larger KuaiRand-1K data also runs on this machine through the
generic loader and pointwise LightGBM path. It measured validation GAUC 0.6853,
nDCG@5 0.6166, and primary 0.6509 on 5,055,984 train and 2,524,980 validation
rows. Its submission passed the generic alignment checker for all 4,132,081
test rows. This is a larger-data execution result, not a claim about the
organizer's hidden-test ranking; no official 1K baseline is asserted here.

## Development tools and APIs

- Python 3.12 and `uv`
- NumPy for data preparation and evaluation
- LightGBM for grouped LambdaRank training
- PyTorch for retained DeepFM/MTL research branches
- OpenAI Responses API with `gpt-5.6-luna`, low reasoning effort, and low text
  verbosity for planning and repair
- Ruff, mypy, and pytest for the automated check loop

## Data and assets

- KuaiRand-Pure starter kit and its organizer-provided train/validation/test
  split
- Organizer `evaluate.py`, baseline, metadata, and submission checker
- No external training data and no benchmark-trained pretrained weights

## Reproduction

```bash
make install
make check
make verify-starter
export OPENAI_API_KEY="..."
UV_NO_CACHE=1 uv run kopibara-agent --run
```

The full controller defaults to the organizer's 50-iteration and 6-hour
limits. Each run writes a manifest, solution-tree node directories, the best
solution, model artifacts, and a checked `submission.csv` under
`runs/autonomous/`.

The lowest-effort high-score local reproduction is `make champion`; the fresh
verified run is under `runs/champion/20260831T142500Z/`.

The optional larger-data check is reproducible with:

```bash
make bonus-1k
```

## Limitations and next improvements

The current generated-code boundary is one self-contained candidate script,
which keeps execution safe and cheap but limits multi-file architectural
changes. The retained DeepFM/MTL and DIN branches are research comparisons;
the winning branch is the simpler history-plus-tree ranker. A future expansion
could allow a small audited candidate workspace for broader model search while
keeping the evaluator and hidden-test boundary immutable. KuaiRand-1k and
KuaiRand-27k are optional bonus benchmarks and are not required for the primary
score.

## Team contribution

Kopibara: autonomous search controller, candidate experiment, safety boundary,
submission validation, experiment logging, and reproducibility tooling.
