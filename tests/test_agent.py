"""Tests for the credential boundary and fixed model choice."""

from kopibara_agent.agent import has_api_key
from kopibara_agent.constants import OPENAI_MODEL


def test_missing_api_key_is_not_ready() -> None:
    assert not has_api_key({})


def test_blank_api_key_is_not_ready() -> None:
    assert not has_api_key({"OPENAI_API_KEY": "   "})


def test_non_empty_api_key_is_ready() -> None:
    assert has_api_key({"OPENAI_API_KEY": "secret"})


def test_uses_requested_luna_model() -> None:
    assert OPENAI_MODEL == "gpt-5.6-luna"
