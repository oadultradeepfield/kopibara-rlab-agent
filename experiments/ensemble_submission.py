"""Average validation-independent model scores into one aligned submission."""

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STARTER_DIRECTORY = PROJECT_ROOT / "kuairand-starter-kit"
sys.path.insert(0, str(STARTER_DIRECTORY))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from evaluate import evaluate  # noqa: E402
from kuairand_dataset import load_dataset  # noqa: E402


def average_scores(score_paths: Sequence[Path]) -> np.ndarray:
    """Average z-normalized finite score arrays with identical row counts."""
    if not score_paths:
        raise ValueError("at least one score file is required")
    arrays = [np.load(path) for path in score_paths]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("each score file must contain a one-dimensional array")
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("score files must have identical lengths")
    normalized = []
    for array in arrays:
        standard_deviation = float(array.std())
        normalized.append(
            (array - array.mean()) / standard_deviation
            if standard_deviation
            else array - array.mean()
        )
    scores = np.mean(np.stack(normalized), axis=0)
    if not np.isfinite(scores).all():
        raise ValueError("averaged scores contain NaN or Inf")
    return scores


def write_submission(
    data_directory: Path,
    score_paths: Sequence[Path],
    output_path: Path,
    valid_score_paths: Sequence[Path] = (),
) -> None:
    """Write averaged scores in the organizer's required row order."""
    rows = load_dataset(data_directory)["test"]
    scores = average_scores(score_paths)
    if len(scores) != len(rows):
        raise ValueError(f"score count {len(scores)} != row count {len(rows)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path.parent / "ensemble_test_scores.npy", scores)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        for row_id, (row, score) in enumerate(zip(rows, scores, strict=True)):
            writer.writerow((row_id, row[1], row[2], float(score)))
    if valid_score_paths:
        valid_rows = load_dataset(data_directory)["valid"]
        valid_scores = average_scores(valid_score_paths)
        if len(valid_scores) != len(valid_rows):
            raise ValueError(
                f"validation score count {len(valid_scores)} != "
                f"row count {len(valid_rows)}"
            )
        metrics = evaluate(
            [row[1] for row in valid_rows],
            [row[6] for row in valid_rows],
            valid_scores,
        )
        (output_path.parent / "metrics.json").write_text(
            json.dumps({"valid": metrics}, indent=2) + "\n",
            encoding="utf-8",
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Average score arrays and write one checked-schema candidate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--valid-scores", nargs="*")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    write_submission(
        Path(args.data_dir),
        [Path(path) for path in args.scores],
        Path(args.output),
        [Path(path) for path in args.valid_scores or []],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
