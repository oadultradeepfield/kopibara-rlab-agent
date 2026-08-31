"""Verify the attached starter kit is present and internally consistent."""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

EXPECTED_FM_TEST_PRIMARY = 0.5946
EXPECTED_FM_VALID_PRIMARY = 0.6016

REQUIRED_FILES = (
    "README.md",
    "ablation_features.py",
    "baseline.py",
    "baseline_scores.json",
    "data.py",
    "evaluate.py",
    "submit.py",
)
EXPECTED_SPLITS = {
    "train": "20220408-20220421",
    "valid": "20220422-20220428",
    "test": "20220429-20220508",
}
EXPECTED_METRICS = ["GAUC", "nDCG@5"]


def require(condition: bool, message: str) -> None:
    """Raise a useful error when one contract check fails."""
    if not condition:
        raise ValueError(message)


def verify_starter_kit(directory: Path) -> None:
    """Check files and published metadata without requiring the dataset."""
    for filename in REQUIRED_FILES:
        require((directory / filename).is_file(), f"missing {filename}")

    with (directory / "baseline_scores.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    require(metadata["dataset"] == "KuaiRand-Pure", "wrong dataset")
    require(metadata["label"] == "long_view", "wrong label")
    require(metadata["metrics"] == EXPECTED_METRICS, "wrong metrics")
    require(metadata["split"] == EXPECTED_SPLITS, "wrong date splits")
    require(
        metadata["convergence_rule"] == {"epsilon": 0.002, "N": 3},
        "wrong convergence rule",
    )
    require(
        metadata["scores"]["fm_official"]["valid"]["primary"]
        == EXPECTED_FM_VALID_PRIMARY,
        "wrong validation baseline",
    )
    require(
        metadata["scores"]["fm_official"]["test"]["primary"]
        == EXPECTED_FM_TEST_PRIMARY,
        "wrong test baseline",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run starter-kit verification."""
    default_directory = Path(__file__).resolve().parents[1] / "kuairand-starter-kit"
    directory = Path(argv[0]) if argv else default_directory
    try:
        verify_starter_kit(directory)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"starter-kit check failed: {error}", file=sys.stderr)
        return 1
    print(f"starter-kit OK: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
