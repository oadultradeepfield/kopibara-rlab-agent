"""Tests for the autonomous controller's pure parsing and stopping logic."""

from pathlib import Path

import pytest

from kopibara_agent.models import CodeEdit, Metrics
from kopibara_agent.planner import apply_code_edits, parse_plan
from kopibara_agent.research_agent import has_converged, write_dashboard_run
from kopibara_agent.runner import (
    build_candidate_command,
    parse_candidate_metrics,
    parse_metrics,
)


def test_parse_metrics_uses_validation_line() -> None:
    metrics = parse_metrics("valid GAUC 0.6697 nDCG@5 0.5366 primary 0.6031")
    assert metrics == Metrics(0.6697, 0.5366, 0.6031)


def test_parse_candidate_metrics_uses_artifact_when_output_is_missing(
    tmp_path: Path,
) -> None:
    (tmp_path / "metrics.json").write_text(
        '{"valid":{"GAUC":0.6999,"nDCG@5":0.5515,"primary":0.6257}}',
        encoding="utf-8",
    )
    assert parse_candidate_metrics("", tmp_path) == Metrics(0.6999, 0.5515, 0.6257)


def test_parse_plan_accepts_fenced_json() -> None:
    plan = parse_plan(
        '```json\n{"action":"edit","parent_id":"000-root",'
        '"title":"pairwise","hypothesis":"align loss",'
        '"stop_rule":"stop if worse","edits":[{"file":"history_lgbm.py",'
        '"old":"before","new":"after"}]}\n```',
        ("000-root",),
    )
    assert plan.parent_id == "000-root"
    assert plan.edits[0] == CodeEdit("history_lgbm.py", "before", "after")


def test_parse_plan_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="edit or stop"):
        parse_plan(
            '{"action":"unknown","parent_id":"000-root","hypothesis":"x",'
            '"stop_rule":"y","edits":[]}',
            ("000-root",),
        )


def test_apply_code_edits_requires_one_exact_match() -> None:
    assert (
        apply_code_edits("alpha beta", [CodeEdit("candidate.py", "beta", "gamma")])
        == "alpha gamma"
    )


def test_apply_code_edits_rejects_ambiguous_match() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        apply_code_edits("alpha alpha", [CodeEdit("candidate.py", "alpha", "beta")])


def test_convergence_requires_three_small_changes() -> None:
    assert not has_converged([0.6015, 0.6020, 0.6025])
    assert has_converged([0.6015, 0.6020, 0.6025, 0.6030])


def test_write_dashboard_run_publishes_the_manifest(tmp_path: Path) -> None:
    path = write_dashboard_run(tmp_path, {"status": "completed"})

    assert path == tmp_path / "frontend" / "public" / "run.json"
    assert '"status": "completed"' in path.read_text(encoding="utf-8")


def test_candidate_command_leaves_model_options_to_the_candidate(
    tmp_path: Path,
) -> None:
    command = build_candidate_command(
        tmp_path / "history_lgbm.py",
        tmp_path / "data",
        tmp_path / "output",
    )

    assert "--objective" not in command
    assert "--feedbacks" not in command
