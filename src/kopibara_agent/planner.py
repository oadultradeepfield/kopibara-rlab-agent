"""Plan validation and prompts for the autonomous research loop."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict

from openai import APIError, OpenAI

from kopibara_agent.agent import ModelAnswer, ask_model_with_usage
from kopibara_agent.constants import (
    CANDIDATE_SCRIPT,
    CONVERGENCE_EPSILON,
    CONVERGENCE_WINDOW,
    DANGEROUS_CODE_TOKENS,
    EDIT_ACTION,
    MAX_CODE_EDITS,
    MAX_PLANNER_ATTEMPTS,
    STOP_ACTION,
)
from kopibara_agent.models import CodeEdit, Metrics, Node, Plan


def parse_plan(raw: str, parent_ids: Sequence[str]) -> Plan:
    """Parse and validate a model plan before it can touch candidate code."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("planner response has no JSON object")
    payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("planner response is not a JSON object")

    action = payload.get("action")
    parent_id = payload.get("parent_id")
    title = payload.get("title", "")
    hypothesis = payload.get("hypothesis", "")
    stop_rule = payload.get("stop_rule", "")
    raw_edits = payload.get("edits", [])
    if action not in (EDIT_ACTION, STOP_ACTION):
        raise ValueError("planner action must be edit or stop")
    if not isinstance(parent_id, str) or parent_id not in parent_ids:
        raise ValueError("planner selected an unknown parent")
    if not all(isinstance(value, str) for value in (title, hypothesis, stop_rule)):
        raise ValueError("planner text fields must be strings")
    if not isinstance(raw_edits, list) or len(raw_edits) > MAX_CODE_EDITS:
        raise ValueError(f"planner must return at most {MAX_CODE_EDITS} edits")

    edits: list[CodeEdit] = []
    for raw_edit in raw_edits:
        if not isinstance(raw_edit, dict):
            raise ValueError("planner edit must be an object")
        file = raw_edit.get("file")
        old = raw_edit.get("old")
        new = raw_edit.get("new")
        if (
            not isinstance(file, str)
            or not isinstance(old, str)
            or not isinstance(new, str)
        ):
            raise ValueError("planner edit fields must be strings")
        if file != CANDIDATE_SCRIPT or not old:
            raise ValueError("planner may edit history_lgbm.py only")
        edits.append(CodeEdit(file, old, new))

    if action == STOP_ACTION and edits:
        raise ValueError("stop action cannot contain edits")
    if action == EDIT_ACTION and not edits:
        raise ValueError("edit action must contain at least one edit")
    return Plan(action, parent_id, title, hypothesis, stop_rule, tuple(edits))


def apply_code_edits(source: str, edits: Sequence[CodeEdit]) -> str:
    """Apply exact, single-match replacements and reject unsafe additions."""
    result = source
    for edit in edits:
        matches = result.count(edit.old)
        if matches != 1:
            raise ValueError(f"edit must match exactly once, found {matches}")
        result = result.replace(edit.old, edit.new, 1)
    for token in DANGEROUS_CODE_TOKENS:
        if token in result and token not in source:
            raise ValueError(f"candidate adds blocked token: {token}")
    return result


def build_planner_prompt(
    baseline: Metrics,
    best: Node,
    nodes: Mapping[str, Node],
    parent: Node,
    source: str,
) -> str:
    """Build the planner prompt from task facts, tree state, and parent code."""
    tree = [
        {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "status": node.status,
            "title": node.title,
            "hypothesis": node.hypothesis,
            "validation_primary": (
                node.metrics.primary if node.metrics is not None else None
            ),
        }
        for node in nodes.values()
    ]
    convergence = (
        f"Organizer convergence: epsilon={CONVERGENCE_EPSILON}, "
        f"consecutive rounds={CONVERGENCE_WINDOW}."
    )
    best_metrics = asdict(best.metrics) if best.metrics else None
    return f"""You are the coding agent inside an autonomous ML research run.
Task: improve KuaiRand-Pure within-user ranking. Train and validation only.
Do not use test labels or hidden-test information. The official evaluator scores
the mean of GAUC and nDCG@5. The official FM baseline uses five fields and
pointwise logloss. Candidate code must remain a runnable Python script.

{convergence}
Official baseline validation: {asdict(baseline)}
Current best node: {best.node_id}, validation: {best_metrics}
Solution tree: {json.dumps(tree)}
Parent node to improve: {parent.node_id}

Research guidance from the starter kit: the organizer already measured static
feature additions and larger FM embeddings as non-improvements. Prioritize
changes that align training with ranking metrics, exploit within-user history
or other released feedback when available, or use an established ranking
objective. Do not repeat a known dead end without a specific new hypothesis.

Return JSON only:
{{"action":"edit"|"stop","parent_id":"{parent.node_id}","title":"short",
"hypothesis":"testable reason","stop_rule":"validation rule",
"edits":[{{"file":"history_lgbm.py","old":"exact existing text",
"new":"replacement text"}}]}}

Rules:
- One focused change; at most {MAX_CODE_EDITS} exact edits.
- Edit history_lgbm.py only. Every old string must occur exactly once.
- Keep CLI arguments and final validation output intact.
- Do not add network, subprocess, shell, filesystem traversal, eval, or exec code.
- If no credible improvement remains, return action=stop with edits=[].

Parent candidate source:
```python
{source}
```
"""


def build_repair_prompt(
    plan: Plan,
    source: str,
    error: str,
    parent_id: str,
) -> str:
    """Build the repair prompt for one failed candidate."""
    return f"""Repair one failed candidate in a controlled ML experiment.
The candidate must remain a standalone Python script with the same CLI and
validation output. Do not use test labels or hidden-test information.
Parent node: {parent_id}
Original hypothesis: {plan.hypothesis}
Failure:
{error[-4000:]}

Return JSON only with this shape:
{{"action":"edit","parent_id":"{parent_id}","title":"repair",
"hypothesis":"repair reason","stop_rule":"stop if repair fails",
"edits":[{{"file":"history_lgbm.py","old":"exact existing text",
"new":"replacement text"}}]}}
Use at most {MAX_CODE_EDITS} exact edits, history_lgbm.py only, no new imports,
no network or subprocess calls.

Current failed source:
```python
{source}
```
"""


def request_plan(
    prompt: str,
    parent_ids: Sequence[str],
    *,
    client: OpenAI | None,
) -> tuple[Plan, int, int, list[str]]:
    """Request valid JSON twice at most and return all recovery errors."""
    errors: list[str] = []
    input_tokens = 0
    output_tokens = 0
    for _ in range(MAX_PLANNER_ATTEMPTS):
        try:
            answer: ModelAnswer = ask_model_with_usage(prompt, client=client)
            input_tokens += answer.input_tokens
            output_tokens += answer.output_tokens
            return (
                parse_plan(answer.text, parent_ids),
                input_tokens,
                output_tokens,
                errors,
            )
        except (APIError, ValueError) as error:
            errors.append(str(error))
    raise RuntimeError("planner failed after retries: " + " | ".join(errors))
