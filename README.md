# Kopibara agent

Kopibara is an autonomous machine-learning research agent. It turns a
benchmark into a bounded code-search loop: propose a measurable hypothesis,
patch a candidate, run it, inspect validation metrics, and retain the best
candidate.

## Architecture

- `src/kopibara_agent/` contains the controller, planner boundary, execution
  policy, recovery path, and run manifest.
- `experiments/` contains runnable candidate models and the KuaiRand adapter.
  The candidates are deliberately ordinary implementations so the controller
  can test different feature, loss, and architecture hypotheses.
- `kuairand-starter-kit/` contains the organizer-provided benchmark contract.
  Benchmark data is downloaded separately and ignored by Git.
- `scripts/` contains local contract checks; `tests/` covers the controller.

## Research loop

1. Read the fixed benchmark contract and validation metric.
2. Ask the configured coding model for a structured, testable hypothesis.
3. Apply only an allowed patch to a candidate experiment.
4. Syntax-check and run the candidate with a timeout.
5. Record the hypothesis, diff, metrics, errors, and recovery events.
6. Keep the best validation node and stop at the configured budget or rule.

The evaluator and controller are outside the generated-code edit boundary.
Candidates receive training data and public validation feedback only. The
KuaiRand adapter masks feedback columns on evaluation rows before returning
them to candidate code.

## Setup

```bash
make install
make check
make verify-starter
```

Place the downloaded KuaiRand data under
`kuairand-starter-kit/KuaiRand-Pure/` or pass a different data directory to an
experiment. The data files and generated runs are intentionally not tracked.

To run the agent, provide credentials in the shell at runtime:

```bash
export OPENAI_API_KEY="..."
UV_NO_CACHE=1 uv run kopibara-agent --run
```

The default client uses the OpenAI Responses API with `gpt-5.6-luna` and low
reasoning effort. Credentials are not stored in the repository.

## Candidate experiments

The useful entry points are:

- `history_lgbm.py`: lagged user, item, author, and user-item feedback
  features with grouped LightGBM ranking.
- `lambdarank.py`: static-feature LightGBM ranking or pointwise fallback.
- `pairwise_fm.py`: a small pairwise Factorization Machine baseline.
- `deepfm_mtl.py`: DeepFM/PLE-style multi-feedback learning comparison.
- `din_ranker.py`: a compact attention model over prior item history.

Each candidate selects models using train and public validation data. Output
files are written to ignored run directories.

## Checks

```bash
make check
make verify-submission
```

The second command validates the local benchmark package and any available
run artifacts. It does not publish anything externally.
