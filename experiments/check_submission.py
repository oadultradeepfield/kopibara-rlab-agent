"""Check a row-aligned submission for either supported KuaiRand layout."""

import argparse
import csv
import math
from collections.abc import Sequence
from pathlib import Path

from kuairand_dataset import load_dataset

EXPECTED_HEADER = ("row_id", "user_id", "video_id", "score")
EXPECTED_COLUMNS = len(EXPECTED_HEADER)


def check_submission(data_directory: Path, submission_path: Path) -> int:
    """Validate schema, alignment, row count, and finite numeric scores."""
    expected_rows = load_dataset(data_directory)["test"]
    with submission_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader, ())) != EXPECTED_HEADER:
            raise ValueError(f"expected header {EXPECTED_HEADER}")
        count = 0
        for row_id, row in enumerate(reader):
            if len(row) != EXPECTED_COLUMNS:
                raise ValueError(f"row {row_id} must have four columns")
            if row[0] != str(row_id):
                raise ValueError(f"row {row_id} has wrong row_id {row[0]!r}")
            if row_id >= len(expected_rows):
                raise ValueError("submission has too many rows")
            expected = expected_rows[row_id]
            if row[1] != expected[1] or row[2] != expected[2]:
                raise ValueError(f"row {row_id} is misaligned")
            try:
                score = float(row[3])
            except ValueError as error:
                raise ValueError(f"row {row_id} score is not numeric") from error
            if not math.isfinite(score):
                raise ValueError(f"row {row_id} score is not finite")
            count += 1
        if count != len(expected_rows):
            raise ValueError(
                f"submission has {count} rows; expected {len(expected_rows)}"
            )
    return len(expected_rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one submission against its data directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--submission", required=True)
    args = parser.parse_args(argv)
    rows = check_submission(Path(args.data_dir), Path(args.submission))
    print(f"submission OK: {rows:,} aligned test rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
