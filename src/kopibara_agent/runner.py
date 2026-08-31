"""Benchmark commands and candidate subprocess execution."""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from kopibara_agent.constants import CANDIDATE_SEED
from kopibara_agent.models import Execution, Metrics, Node

METRIC_PATTERN = re.compile(
    r"valid\s+GAUC\s+(?P<gauc>[0-9.]+)\s+"
    r"nDCG@5\s+(?P<ndcg>[0-9.]+)\s+primary\s+(?P<primary>[0-9.]+)"
)


def build_baseline_command(root: Path, data_directory: Path) -> tuple[str, ...]:
    """Build the fixed organizer-baseline command."""
    return (
        sys.executable,
        str(root / "kuairand-starter-kit" / "baseline.py"),
        "--data_dir",
        str(data_directory),
        "--model",
        "fm",
        "--seed",
        "0",
    )


def load_official_baseline(root: Path) -> Metrics:
    """Read the organizer-published validation reference from the kit."""
    metadata_path = root / "kuairand-starter-kit" / "baseline_scores.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    score = metadata["scores"]["fm_official"]["valid"]
    return Metrics(
        float(score["GAUC"]),
        float(score["nDCG@5"]),
        float(score["primary"]),
    )


def parse_metrics(output: str) -> Metrics:
    """Parse only the validation line emitted by a candidate."""
    matches = list(METRIC_PATTERN.finditer(output))
    if not matches:
        raise ValueError("no validation metrics found in candidate output")
    match = matches[-1]
    return Metrics(
        gauc=float(match["gauc"]),
        ndcg_at_5=float(match["ndcg"]),
        primary=float(match["primary"]),
    )


def parse_candidate_metrics(output: str, output_directory: Path) -> Metrics:
    """Parse candidate stdout, falling back to its own metrics artifact."""
    try:
        return parse_metrics(output)
    except ValueError as output_error:
        metrics_path = output_directory / "metrics.json"
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            valid = payload["valid"]
            return Metrics(
                gauc=float(valid["GAUC"]),
                ndcg_at_5=float(valid["nDCG@5"]),
                primary=float(valid["primary"]),
            )
        except (KeyError, OSError, TypeError, ValueError) as artifact_error:
            raise ValueError(
                f"{output_error}; metrics artifact unavailable or invalid"
            ) from artifact_error


def run_command(
    command: Sequence[str],
    *,
    root: Path,
    timeout_seconds: float,
    environment: Mapping[str, str],
) -> Execution:
    """Run a fixed command without shell interpretation."""
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        cwd=root,
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"exit code {completed.returncode}: {output[-4000:]}")
    return Execution(output, time.monotonic() - started)


def build_environment(root: Path) -> dict[str, str]:
    """Give candidates project location without forwarding the API key."""
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment["KUAI_PROJECT_ROOT"] = str(root)
    return environment


def build_candidate_command(
    code_path: Path,
    data_directory: Path,
    output_directory: Path,
) -> tuple[str, ...]:
    """Build the only candidate command the controller can execute."""
    return (
        sys.executable,
        str(code_path),
        "--data-dir",
        str(data_directory),
        "--objective",
        "lambdarank",
        "--feedbacks",
        "all",
        "--seed",
        str(CANDIDATE_SEED),
        "--output-dir",
        str(output_directory),
    )


def run_candidate(
    code_path: Path,
    data_directory: Path,
    output_directory: Path,
    *,
    root: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[Execution, tuple[str, ...]]:
    """Compile and execute one candidate with a bounded timeout."""
    compile_command = (sys.executable, "-m", "py_compile", str(code_path))
    compile_execution = run_command(
        compile_command,
        root=root,
        timeout_seconds=timeout_seconds,
        environment=environment,
    )
    command = build_candidate_command(code_path, data_directory, output_directory)
    execution = run_command(
        command,
        root=root,
        timeout_seconds=timeout_seconds,
        environment=environment,
    )
    return (
        Execution(
            execution.output,
            compile_execution.seconds + execution.seconds,
        ),
        command,
    )


def copy_best(node: Node, run_directory: Path) -> tuple[Path, Path | None]:
    """Copy the best code and available model artifacts to stable paths."""
    best_code = run_directory / "best_solution.py"
    shutil.copy2(node.code_path, best_code)
    if node.output_directory is None:
        return best_code, None
    best_output = run_directory / "best_output"
    if best_output.exists():
        shutil.rmtree(best_output)
    shutil.copytree(node.output_directory, best_output)
    return best_code, best_output


def finalize_submission(
    best: Node,
    run_directory: Path,
    *,
    root: Path,
    data_directory: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> Path | None:
    """Create and validate the final aligned submission."""
    _, best_output = copy_best(best, run_directory)
    submission_path = run_directory / "submission.csv"
    if best_output is None:
        command = (
            sys.executable,
            str(root / "kuairand-starter-kit" / "submit.py"),
            "--data_dir",
            str(data_directory),
            "--split",
            "test",
            "--make",
            str(submission_path),
        )
    else:
        command = (
            sys.executable,
            str(root / "experiments" / "write_submission.py"),
            "--data-dir",
            str(data_directory),
            "--scores",
            str(best_output / "test_scores.npy"),
            "--output",
            str(submission_path),
        )
    run_command(
        command,
        root=root,
        timeout_seconds=timeout_seconds,
        environment=environment,
    )
    check_command = (
        sys.executable,
        str(root / "kuairand-starter-kit" / "submit.py"),
        "--data_dir",
        str(data_directory),
        "--split",
        "test",
        "--check",
        str(submission_path),
    )
    run_command(
        check_command,
        root=root,
        timeout_seconds=timeout_seconds,
        environment=environment,
    )
    return submission_path
