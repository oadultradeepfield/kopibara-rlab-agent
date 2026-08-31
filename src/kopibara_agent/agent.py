"""Small OpenAI Responses API adapter."""

from collections.abc import Mapping
from dataclasses import dataclass

from openai import OpenAI

from kopibara_agent.constants import (
    OPENAI_API_KEY_ENVIRONMENT,
    OPENAI_MODEL,
    OPENAI_REASONING_EFFORT,
    OPENAI_TEXT_VERBOSITY,
)


@dataclass(frozen=True, slots=True)
class ModelAnswer:
    """Text and usage returned by one model request."""

    text: str
    input_tokens: int
    output_tokens: int


def has_api_key(environment: Mapping[str, str]) -> bool:
    """Return whether the configured API key is non-empty."""
    return bool(environment.get(OPENAI_API_KEY_ENVIRONMENT, "").strip())


def ask_model(prompt: str, *, client: OpenAI | None = None) -> str:
    """Ask Luna one question through the Responses API."""
    return ask_model_with_usage(prompt, client=client).text


def ask_model_with_usage(prompt: str, *, client: OpenAI | None = None) -> ModelAnswer:
    """Ask Luna and retain token accounting for the run manifest."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    active_client = client if client is not None else OpenAI()
    response = active_client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": OPENAI_REASONING_EFFORT},
        text={"verbosity": OPENAI_TEXT_VERBOSITY},
        input=prompt,
    )
    usage = response.usage
    return ModelAnswer(
        text=response.output_text,
        input_tokens=usage.input_tokens if usage is not None else 0,
        output_tokens=usage.output_tokens if usage is not None else 0,
    )
