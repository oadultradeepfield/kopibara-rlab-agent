"""Stable configuration for the first agent shell."""

from typing import Final, Literal

OPENAI_API_KEY_ENVIRONMENT: Final[str] = "OPENAI_API_KEY"
OPENAI_MODEL: Final[str] = "gpt-5.6-luna"
OPENAI_REASONING_EFFORT: Final[Literal["low"]] = "low"
OPENAI_TEXT_VERBOSITY: Final[Literal["low"]] = "low"
