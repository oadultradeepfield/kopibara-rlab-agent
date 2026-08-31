"""Write and align a submission from candidate scores."""

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STARTER_DIRECTORY = PROJECT_ROOT / "kuairand-starter-kit"
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from kuairand_dataset import load_dataset  # noqa: E402


def write_submission(
    data_directory: Path,
    scores_path: Path,
    output_path: Path,
) -> None:
    """Write scores in the organizer's row-aligned CSV schema."""
    rows = load_dataset(data_directory)["test"]
    scores = np.load(scores_path)
    if len(scores) != len(rows):
        raise ValueError(f"score count {len(scores)} != row count {len(rows)}")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        for row_id, (row, score) in enumerate(zip(rows, scores, strict=True)):
            writer.writerow((row_id, row[1], row[2], float(score)))


def main(argv: Sequence[str] | None = None) -> int:
    """Write one test-split submission."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    write_submission(Path(args.data_dir), Path(args.scores), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
