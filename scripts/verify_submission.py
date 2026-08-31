"""Verify the local submission package and its latest run artifact."""

import argparse
import json
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

EXPECTED_PROJECT_NAME = "kopibara-agent"
EXPECTED_COMMAND = "kopibara-agent"
REQUIRED_PROJECT_FILES = (
    "README.md",
    "SUBMISSION_DESCRIPTION.md",
    "SUBMISSION_CHECKLIST.md",
    "pyproject.toml",
    "uv.lock",
    "Makefile",
    "src/kopibara_agent/cli.py",
    "experiments/pairwise_fm.py",
    "experiments/history_lgbm.py",
    "experiments/ensemble_submission.py",
    "experiments/run_winning_pipeline.py",
    "scripts/verify_starter.py",
)


def require(condition: bool, message: str) -> None:
    """Raise a useful error when a local requirement is missing."""
    if not condition:
        raise ValueError(message)


def find_latest_run(root: Path) -> Path:
    """Return the newest autonomous run directory."""
    candidates = sorted((root / "runs" / "autonomous").glob("20*"))
    if not candidates:
        raise ValueError("no autonomous run directory found")
    return candidates[-1]


def find_latest_champion(root: Path) -> Path | None:
    """Return the newest local high-score pipeline run, when present."""
    candidates = sorted((root / "runs" / "champion").glob("20*"))
    return candidates[-1] if candidates else None


def check_submission(root: Path, run_directory: Path) -> None:
    """Check code naming, logs, outputs, and the official row validator."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    require(project["project"]["name"] == EXPECTED_PROJECT_NAME, "wrong project name")
    require(
        EXPECTED_COMMAND in project["project"]["scripts"],
        "missing Kopibara command",
    )
    for relative_path in REQUIRED_PROJECT_FILES:
        require((root / relative_path).is_file(), f"missing {relative_path}")
    require(not (root / "src/kuairand_agent").exists(), "old package remains")

    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["status"] == "completed", "run is not completed")
    require(manifest["manual_interventions"] == 0, "manual intervention count missing")
    require(manifest["hidden_test_access"] is False, "hidden-test boundary failed")
    require(manifest["total_input_tokens"] >= 0, "input token count missing")
    require(manifest["total_output_tokens"] >= 0, "output token count missing")
    submission = Path(manifest["final_submission"])
    submission_path = submission if submission.is_absolute() else root / submission
    require(submission_path.is_file(), "final submission is missing")

    iteration_logs = sorted(run_directory.glob("[0-9]*.json"))
    require(iteration_logs, "iteration logs are missing")
    for log_path in iteration_logs:
        record = json.loads(log_path.read_text(encoding="utf-8"))
        require("iteration" in record, f"iteration missing in {log_path.name}")
        if record["iteration"] > 0 and "status" in record:
            require(
                "code_diff" in record or "plan" in record,
                f"evidence missing in {log_path.name}",
            )

    command = (
        sys.executable,
        str(root / "kuairand-starter-kit" / "submit.py"),
        "--data_dir",
        str(root / "kuairand-starter-kit/KuaiRand-Pure/data"),
        "--split",
        "test",
        "--check",
        str(submission_path),
    )
    result = subprocess.run(
        command, cwd=root, capture_output=True, text=True, check=False
    )
    require(result.returncode == 0, result.stdout + result.stderr)


def check_champion(root: Path, run_directory: Path) -> None:
    """Check the current high-score pipeline manifest and submission."""
    manifest = json.loads((run_directory / "manifest.json").read_text())
    require(manifest["status"] == "completed", "champion run is not completed")
    require(
        manifest["manual_interventions"] == 0, "champion intervention count missing"
    )
    require(manifest["hidden_test_access"] is False, "champion used hidden-test data")
    submission = Path(manifest["final_submission"])
    submission_path = submission if submission.is_absolute() else root / submission
    require(submission_path.is_file(), "champion submission is missing")
    command = (
        sys.executable,
        str(root / "kuairand-starter-kit" / "submit.py"),
        "--data_dir",
        str(root / "kuairand-starter-kit/KuaiRand-Pure/data"),
        "--split",
        "test",
        "--check",
        str(submission_path),
    )
    result = subprocess.run(
        command, cwd=root, capture_output=True, text=True, check=False
    )
    require(result.returncode == 0, result.stdout + result.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """Run local submission checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    run_directory = (
        Path(args.run_directory).resolve()
        if args.run_directory
        else find_latest_run(root)
    )
    try:
        check_submission(root, run_directory)
        champion = find_latest_champion(root)
        if champion is not None:
            check_champion(root, champion)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"submission check failed: {error}", file=sys.stderr)
        return 1
    print(f"local submission package OK: {run_directory}")
    print("local-only verification complete; external publishing was not checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
