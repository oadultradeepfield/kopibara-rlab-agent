# Kopibara autonomous research agent

An AIDE-style autonomous ML research agent for the KuaiRand challenge. The
model is only the planner: the product owns the experiment loop, executes
candidate code, recovers from failures, chooses the best validation node, and
produces a checked submission.

## Setup

```bash
make install
make check
```

The starter kit is copied to [`kuairand-starter-kit/`](kuairand-starter-kit/),
including the local KuaiRand-Pure data when it has been downloaded.
Run its contract check with:

```bash
make verify-starter
```

The attached zip contains the official code and metadata. Keep benchmark data
under `kuairand-starter-kit/KuaiRand-Pure/`; it is ignored by Git.

## Winning local direction

The strongest measured Pure direction is a grouped LightGBM LambdaRank model
with leakage-safe historical target features. For every impression, the
feature builder uses only earlier train or public-validation feedback for the
same user, video, author, and user-video pair. It combines these rates and
counts with the released categorical metadata and all seven binary feedback
signals. Test rows never update the history state.

Three seeds are z-normalized and ensembled before writing the final CSV:

```bash
make history-ensemble
```

The verified local validation result is GAUC 0.7007 / nDCG@5 0.5520 / primary
0.6264, against the official 0.6674 / 0.5357 / 0.6016. That is a +0.0248
validation-primary delta. The fresh run and its evidence are under
`runs/champion/20260831T142500Z/`. This is a validation result; the
organizer's hidden-test score is still unavailable locally.

On macOS, LightGBM may also require the system OpenMP runtime:

```bash
brew install libomp
```

## Autonomous run

Credentials are never stored in this project. Set the key only in the shell
when ready:

```bash
export OPENAI_API_KEY="..."
uv run kopibara-agent --check-api
UV_NO_CACHE=1 uv run kopibara-agent --run
```

The run reproduces the official FM baseline, starts from the measured
historical-feature ranker, and then runs a best-first solution-tree search. On each iteration
Luna proposes a structured hypothesis and exact patch against the selected
parent. The controller syntax-checks and executes the child without shell
expansion, scores validation only, and gives a failed child one automatic
repair attempt. Better children become the current best. The run stops at the
organizer's 50-iteration / 6-hour limits or its epsilon=0.002, N=3 convergence
rule.

Every iteration records the hypothesis, parent and child code diff, command,
validation metrics, token usage, errors, recovery events, and selection result.
The run retains `best_solution.py`, its model artifacts, and an aligned
`submission.csv`; the submission is checked with the untouched starter
`submit.py`. The manifest reports total LLM tokens, wall-clock time, iteration
count, manual interventions, hidden-test access declaration, and the complete
solution tree.

The safe generated-code boundary is deliberately explicit: Luna can patch
`experiments/history_lgbm.py` only. That single runnable file contains the
history construction, model, grouped training loop, validation checkpointing,
and prediction output, so the agent can explore algorithmic changes without
letting generated code alter the evaluator, submission checker, or controller.

For a short smoke run before spending API budget:

```bash
UV_NO_CACHE=1 uv run kopibara-agent --run --max-iterations 3 --wall-clock-hours 0.25
```

The adapter uses the OpenAI Responses API with model `gpt-5.6-luna`, low
reasoning effort, and low text verbosity. It makes no request during import or
startup checks. [AIDE's paper](https://arxiv.org/abs/2502.13138) describes the
code-space tree-search pattern this controller follows.

## Research branches

The repository also retains the smaller pairwise FM, DeepFM/MTL, DIN history,
and plain LambdaRank branches used to falsify weaker directions:

```bash
make pairwise
```

This reports validation metrics only. It still writes test predictions for the
submission artifact, but it never evaluates or uses test labels.

The starter kit's `evaluate.py` remains the evaluation authority. The current
safe edit boundary is `experiments/history_lgbm.py`; the controller applies the
same parser, syntax check, timeout, and failure-repair path to it.

The optional KuaiRand-1K archive is supported by the same generic loader and
submission writer. `make bonus-1k-lgbm` runs the larger-data-compatible
pointwise LightGBM path, writes a submission, and validates its 4M-row
alignment locally. The grouped LambdaRank objective is not compatible with a
measured 1K user query containing 11,704 rows because LightGBM caps a query at
10,000 rows.
