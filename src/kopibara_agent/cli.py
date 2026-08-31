"""Command-line entry point for Kopibara's agent."""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from kopibara_agent.agent import ask_model, has_api_key
from kopibara_agent.constants import OPENAI_MODEL
from kopibara_agent.research_agent import run_agent


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Run the Kopibara agent.")
    parser.add_argument("--check-api", action="store_true")
    parser.add_argument("--prompt")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--data-dir", default="kuairand-starter-kit/KuaiRand-Pure/data")
    parser.add_argument("--log-dir", default="runs/autonomous")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--wall-clock-hours", type=float, default=6.0)
    parser.add_argument("--candidate-timeout-seconds", type=float, default=900.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a configuration check or one model request."""
    args = build_parser().parse_args(argv)
    if args.check_api:
        state = "set" if has_api_key(os.environ) else "missing"
        print(f"model={OPENAI_MODEL} api_key={state}")
        return 0
    if args.run:
        if not has_api_key(os.environ):
            raise SystemExit("set OPENAI_API_KEY before --run")
        summary = run_agent(
            Path.cwd(),
            Path(args.data_dir),
            Path(args.log_dir),
            max_iterations=args.max_iterations,
            wall_clock_seconds=args.wall_clock_hours * 60 * 60,
            timeout_seconds=args.candidate_timeout_seconds,
        )
        print(
            f"best validation GAUC={summary.best_validation.gauc:.4f} "
            f"nDCG@5={summary.best_validation.ndcg_at_5:.4f} "
            f"primary={summary.best_validation.primary:.4f} "
            f"iterations={summary.iterations} "
            f"tokens={summary.total_input_tokens + summary.total_output_tokens} "
            f"logs={summary.run_directory} "
            f"submission={summary.final_submission}"
        )
        return 0
    if args.prompt is None:
        raise SystemExit("provide --prompt, --run, or use --check-api")
    print(ask_model(args.prompt))
    return 0
