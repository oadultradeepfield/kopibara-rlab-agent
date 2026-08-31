"""Stable configuration for the first agent shell."""

from pathlib import Path
from typing import Final, Literal

OPENAI_API_KEY_ENVIRONMENT: Final[str] = "OPENAI_API_KEY"
OPENAI_MODEL: Final[str] = "gpt-5.6-luna"
OPENAI_REASONING_EFFORT: Final[Literal["low"]] = "low"
OPENAI_TEXT_VERBOSITY: Final[Literal["low"]] = "low"
CANDIDATE_SEED: Final[int] = 0
CANDIDATE_SOURCE: Final[Path] = Path("experiments/history_lgbm.py")
CANDIDATE_SCRIPT: Final[str] = "history_lgbm.py"
CONVERGENCE_EPSILON: Final[float] = 0.002
CONVERGENCE_WINDOW: Final[int] = 3
MAX_CODE_EDITS: Final[int] = 4
MAX_PLANNER_ATTEMPTS: Final[int] = 2
STOP_ACTION: Final[str] = "stop"
EDIT_ACTION: Final[str] = "edit"
DANGEROUS_CODE_TOKENS: Final[tuple[str, ...]] = (
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "httpx",
    "openai",
    "os.system",
    "eval(",
    "exec(",
    "__import__",
)
