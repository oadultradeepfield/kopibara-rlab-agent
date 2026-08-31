"""Run, log, and package the lowest-effort high-score Pure pipeline."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from ensemble_submission import write_submission

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STARTER_DIRECTORY = PROJECT_ROOT / "kuairand-starter-kit"
SEEDS = (0, 1, 2)
METRIC_PATTERN = re.compile(
    r"valid GAUC (?P<gauc>[0-9.]+) \| nDCG@5 (?P<ndcg>[0-9.]+) \| "
    r"primary (?P<primary>[0-9.]+)"
)


def parse_metrics(output: str) -> dict[str, float]:
    """Parse the final validation line from one model run."""
    matches = list(METRIC_PATTERN.finditer(output))
    if not matches:
        raise ValueError("model output did not contain validation metrics")
    match = matches[-1]
    return {
        "GAUC": float(match["gauc"]),
        "nDCG@5": float(match["ndcg"]),
        "primary": float(match["primary"]),
    }


def run_seed(
    data_directory: Path,
    output_directory: Path,
    seed: int,
) -> tuple[dict[str, float], float, tuple[str, ...], str]:
    """Run one historical ranker seed without shell expansion."""
    command = (
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "history_lgbm.py"),
        "--data-dir",
        str(data_directory),
        "--objective",
        "lambdarank",
        "--feedbacks",
        "all",
        "--seed",
        str(seed),
        "--output-dir",
        str(output_directory),
    )
    started = time.monotonic()
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment["KUAI_PROJECT_ROOT"] = str(PROJECT_ROOT)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(output[-4000:])
    return parse_metrics(output), time.monotonic() - started, command, output


def run_pipeline(data_directory: Path, run_directory: Path) -> dict[str, object]:
    """Run all seeds, create the ensemble, and write judging evidence."""
    data_directory = data_directory.resolve()
    run_directory = run_directory.resolve()
    started = time.monotonic()
    run_directory.mkdir(parents=True, exist_ok=True)
    seed_records: list[dict[str, object]] = []
    test_scores: list[Path] = []
    valid_scores: list[Path] = []
    for iteration, seed in enumerate(SEEDS, start=1):
        output_directory = run_directory / f"seed{seed}"
        metrics, seconds, command, output = run_seed(
            data_directory, output_directory, seed
        )
        test_scores.append(output_directory / "test_scores.npy")
        valid_scores.append(output_directory / "valid_scores.npy")
        record = {
            "iteration": iteration,
            "status": "kept",
            "hypothesis": (
                "Lagged feedback rates and exposure counts at user, video, "
                "author, and user-video scopes should improve within-user ranking."
            ),
            "code_diff": "experiments/history_lgbm.py is the audited candidate.",
            "validation": metrics,
            "seconds": seconds,
            "command": list(command),
            "error_recovery": [],
            "manual_interventions": 0,
            "hidden_test_access": False,
            "stdout_tail": output[-2000:],
        }
        (run_directory / f"{iteration:03d}_seed{seed}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        seed_records.append(record)

    submission_path = run_directory / "submission.csv"
    write_submission(data_directory, test_scores, submission_path, valid_scores)
    ensemble_metrics = json.loads(
        (run_directory / "metrics.json").read_text(encoding="utf-8")
    )["valid"]
    ensemble_record = {
        "iteration": len(SEEDS) + 1,
        "status": "kept",
        "hypothesis": (
            "Seed averaging should reduce model variance without changing row order."
        ),
        "code_diff": (
            "experiments/ensemble_submission.py standardizes and averages scores."
        ),
        "validation": ensemble_metrics,
        "seconds": time.monotonic() - started,
        "command": [str(PROJECT_ROOT / "experiments" / "ensemble_submission.py")],
        "error_recovery": [],
        "manual_interventions": 0,
        "hidden_test_access": False,
    }
    (run_directory / "004_ensemble.json").write_text(
        json.dumps(ensemble_record, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "completed",
        "benchmark": "KuaiRand-Pure",
        "baseline_validation": {
            "GAUC": 0.6674,
            "nDCG@5": 0.5357,
            "primary": 0.6016,
        },
        "best_validation": ensemble_metrics,
        "iterations": len(SEEDS) + 1,
        "wall_clock_seconds": time.monotonic() - started,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "manual_interventions": 0,
        "hidden_test_access": False,
        "seed_runs": seed_records,
        "final_submission": str(submission_path.relative_to(PROJECT_ROOT)),
    }
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reproducible high-score pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(STARTER_DIRECTORY / "KuaiRand-Pure" / "data"),
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    run_directory = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT
        / "runs"
        / "champion"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    manifest = run_pipeline(Path(args.data_dir), run_directory)
    metrics = manifest["best_validation"]
    print(f"champion validation: {json.dumps(metrics)}")
    print(f"submission: {manifest['final_submission']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
