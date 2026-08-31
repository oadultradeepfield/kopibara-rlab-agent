"""AIDE-style autonomous code search for the KuaiRand benchmark."""

import difflib
import json
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from openai import APIError, OpenAI

from kopibara_agent.constants import (
    CANDIDATE_SCRIPT,
    CANDIDATE_SOURCE,
    CONVERGENCE_EPSILON,
    CONVERGENCE_WINDOW,
    STOP_ACTION,
)
from kopibara_agent.models import (
    AgentSummary,
    CandidateAttempt,
    IterationOutcome,
    Node,
    Plan,
    SearchState,
)
from kopibara_agent.planner import (
    apply_code_edits,
    build_planner_prompt,
    build_repair_prompt,
    request_plan,
)
from kopibara_agent.runner import (
    build_baseline_command,
    build_candidate_command,
    build_environment,
    copy_best,
    finalize_submission,
    load_official_baseline,
    parse_candidate_metrics,
    parse_metrics,
    run_candidate,
    run_command,
)


def has_converged(
    scores: Sequence[float],
    *,
    epsilon: float = CONVERGENCE_EPSILON,
    window: int = CONVERGENCE_WINDOW,
) -> bool:
    """Apply the organizer's N consecutive small-improvement rule."""
    if len(scores) < window + 1:
        return False
    recent = scores[-(window + 1) :]
    return all(recent[index + 1] - recent[index] <= epsilon for index in range(window))


def select_parent(nodes: Mapping[str, Node]) -> Node:
    """Select the highest-scoring measured node for best-first search."""
    measured = [
        node
        for node in nodes.values()
        if node.metrics is not None and node.status != "recovered_failure"
    ]
    if not measured:
        raise ValueError("solution tree has no measured node")
    return max(
        measured,
        key=lambda node: node.metrics.primary if node.metrics else -1.0,
    )


def write_log(path: Path, record: Mapping[str, object]) -> None:
    """Write a complete iteration record."""
    path.write_text(
        json.dumps(record, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def attempt_candidate(
    plan: Plan,
    parent: Node,
    parent_source: str,
    child_code: Path,
    child_output: Path,
    *,
    root: Path,
    data_directory: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    client: OpenAI | None,
) -> CandidateAttempt:
    """Apply, run, and repair one generated child candidate."""
    recovery_events: list[str] = []
    input_tokens = 0
    output_tokens = 0
    source = parent_source
    command = build_candidate_command(child_code, data_directory, child_output)
    try:
        source = apply_code_edits(parent_source, plan.edits)
        child_code.write_text(source, encoding="utf-8")
        execution, command = run_candidate(
            child_code,
            data_directory,
            child_output,
            root=root,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
        return CandidateAttempt(
            source,
            parse_candidate_metrics(execution.output, child_output),
            execution,
            command,
            tuple(recovery_events),
            input_tokens,
            output_tokens,
        )
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        recovery_events.append(str(error))

    repair_prompt = build_repair_prompt(
        plan,
        parent_source,
        recovery_events[-1],
        parent.node_id,
    )
    try:
        repair_plan, repair_input, repair_output, repair_errors = request_plan(
            repair_prompt,
            (parent.node_id,),
            client=client,
        )
        input_tokens += repair_input
        output_tokens += repair_output
        source = apply_code_edits(parent_source, repair_plan.edits)
        child_code.write_text(source, encoding="utf-8")
        execution, command = run_candidate(
            child_code,
            data_directory,
            child_output,
            root=root,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
        recovery_events.extend(repair_errors)
        recovery_events.append("candidate repaired and rerun")
        return CandidateAttempt(
            source,
            parse_candidate_metrics(execution.output, child_output),
            execution,
            command,
            tuple(recovery_events),
            input_tokens,
            output_tokens,
        )
    except (
        APIError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.TimeoutExpired,
    ) as error:
        recovery_events.append(f"repair failed: {error}")
        return CandidateAttempt(
            source,
            None,
            None,
            command,
            tuple(recovery_events),
            input_tokens,
            output_tokens,
        )


def initialize_search(
    root: Path,
    data_directory: Path,
    run_directory: Path,
    node_directory: Path,
    *,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> SearchState:
    """Verify the official baseline and measure the initial candidate node."""
    baseline_execution = run_command(
        build_baseline_command(root, data_directory),
        root=root,
        timeout_seconds=timeout_seconds,
        environment=environment,
    )
    runtime_baseline = parse_metrics(baseline_execution.output)
    baseline = load_official_baseline(root)
    root_directory = node_directory / "000-root"
    root_code = root_directory / CANDIDATE_SCRIPT
    root_output = root_directory / "output"
    root_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / CANDIDATE_SOURCE, root_code)
    root_execution, root_command = run_candidate(
        root_code,
        data_directory,
        root_output,
        root=root,
        environment=environment,
        timeout_seconds=timeout_seconds,
    )
    root_metrics = parse_candidate_metrics(root_execution.output, root_output)
    root_node = Node("000-root", None, root_code, root_output, root_metrics, "kept", 0)
    best = (
        root_node
        if root_metrics.primary > baseline.primary
        else Node(
            "baseline",
            None,
            root / "kuairand-starter-kit" / "baseline.py",
            None,
            baseline,
            "baseline",
            0,
        )
    )
    write_log(
        run_directory / "000_root.json",
        {
            "iteration": 0,
            "status": "kept",
            "node_id": root_node.node_id,
            "parent_id": None,
            "baseline_validation": asdict(baseline),
            "runtime_baseline_validation": asdict(runtime_baseline),
            "validation": asdict(root_metrics),
            "seconds": baseline_execution.seconds + root_execution.seconds,
            "command": list(root_command),
            "code_diff": "seed candidate copied from experiments/history_lgbm.py",
            "manual_interventions": 0,
            "hidden_test_access": False,
        },
    )
    return SearchState(
        baseline,
        runtime_baseline,
        best,
        {root_node.node_id: root_node},
        [baseline.primary, root_metrics.primary],
    )


def run_search_iteration(
    iteration: int,
    state: SearchState,
    run_directory: Path,
    node_directory: Path,
    *,
    root: Path,
    data_directory: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    client: OpenAI | None,
) -> IterationOutcome:
    """Plan, execute, recover, score, and log one tree-search child."""
    parent = select_parent(state.nodes)
    parent_source = parent.code_path.read_text(encoding="utf-8")
    prompt = build_planner_prompt(
        state.baseline,
        state.best,
        state.nodes,
        parent,
        parent_source,
    )
    try:
        plan, used_input, used_output, planner_errors = request_plan(
            prompt,
            (parent.node_id,),
            client=client,
        )
    except RuntimeError as error:
        write_log(
            run_directory / f"{iteration:03d}_planner_failure.json",
            {
                "iteration": iteration,
                "status": "recovered_planner_failure",
                "error": str(error),
                "manual_interventions": 0,
                "hidden_test_access": False,
            },
        )
        return IterationOutcome(False, 0, 0, "planner_failure")
    if plan.action == STOP_ACTION:
        write_log(
            run_directory / f"{iteration:03d}_stop.json",
            {
                "iteration": iteration,
                "status": "stopped_by_agent",
                "plan": asdict(plan),
                "planner_recovery": planner_errors,
                "manual_interventions": 0,
                "hidden_test_access": False,
            },
        )
        return IterationOutcome(False, used_input, used_output, "agent_stop")

    node_id = f"{iteration:03d}-child"
    child_directory = node_directory / node_id
    child_directory.mkdir(parents=True, exist_ok=True)
    child_code = child_directory / CANDIDATE_SCRIPT
    child_output = child_directory / "output"
    attempt = attempt_candidate(
        plan,
        parent,
        parent_source,
        child_code,
        child_output,
        root=root,
        data_directory=data_directory,
        environment=environment,
        timeout_seconds=timeout_seconds,
        client=client,
    )
    node_status = "evaluated" if attempt.metrics is not None else "recovered_failure"
    node = Node(
        node_id,
        parent.node_id,
        child_code,
        child_output if attempt.metrics is not None else None,
        attempt.metrics,
        node_status,
        parent.depth + 1,
        plan.title,
        plan.hypothesis,
    )
    state.nodes[node_id] = node
    is_best = attempt.metrics is not None and (
        state.best.metrics is None
        or attempt.metrics.primary > state.best.metrics.primary
    )
    if is_best:
        state.best = node
        copy_best(node, run_directory)
    fallback_score = state.best.metrics.primary if state.best.metrics else -1.0
    state.scores.append(
        attempt.metrics.primary if attempt.metrics is not None else fallback_score
    )
    diff = "".join(
        difflib.unified_diff(
            parent_source.splitlines(keepends=True),
            attempt.source.splitlines(keepends=True),
            fromfile=parent.node_id,
            tofile=node_id,
        )
    )
    write_log(
        run_directory / f"{iteration:03d}_{node_id}.json",
        {
            "iteration": iteration,
            "node_id": node_id,
            "parent_id": parent.node_id,
            "plan": asdict(plan),
            "status": "kept" if is_best else node.status,
            "validation": asdict(attempt.metrics) if attempt.metrics else None,
            "seconds": attempt.execution.seconds if attempt.execution else None,
            "command": list(attempt.command),
            "code_diff": diff,
            "recovery_events": attempt.recovery_events,
            "planner_recovery": planner_errors,
            "input_tokens": attempt.input_tokens + used_input,
            "output_tokens": attempt.output_tokens + used_output,
            "manual_interventions": 0,
            "hidden_test_access": False,
        },
    )
    return IterationOutcome(
        True,
        used_input + attempt.input_tokens,
        used_output + attempt.output_tokens,
        None,
    )


def build_manifest(
    state: SearchState,
    *,
    root: Path,
    stopped_reason: str,
    attempted: int,
    max_iterations: int,
    elapsed_seconds: float,
    input_tokens: int,
    output_tokens: int,
    final_submission: Path | None,
) -> dict[str, object]:
    """Build the reproducibility and judging manifest."""
    return {
        "status": "completed",
        "stopped_reason": stopped_reason,
        "baseline_validation": asdict(state.baseline),
        "runtime_baseline_validation": asdict(state.runtime_baseline),
        "best_validation": asdict(state.best.metrics) if state.best.metrics else None,
        "best_node_id": state.best.node_id,
        "iterations": attempted,
        "max_iterations": max_iterations,
        "wall_clock_seconds": elapsed_seconds,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "manual_interventions": 0,
        "hidden_test_access": False,
        "nodes": [
            {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "status": node.status,
                "title": node.title,
                "hypothesis": node.hypothesis,
                "validation": asdict(node.metrics) if node.metrics else None,
            }
            for node in state.nodes.values()
        ],
        "final_submission": (
            str(final_submission.relative_to(root))
            if final_submission is not None
            else None
        ),
    }


def run_agent(
    root: Path,
    data_directory: Path,
    log_directory: Path,
    *,
    max_iterations: int = 50,
    wall_clock_seconds: float = 6 * 60 * 60,
    timeout_seconds: float = 900.0,
    client: OpenAI | None = None,
) -> AgentSummary:
    """Run baseline, tree search, code generation, repair, and submission."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    root = root.resolve()
    data_directory = data_directory.resolve()
    run_directory = log_directory.resolve() / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    node_directory = run_directory / "nodes"
    node_directory.mkdir(parents=True, exist_ok=True)
    environment = build_environment(root)
    started = time.monotonic()
    input_tokens = 0
    output_tokens = 0
    stopped_reason = "iteration_cap"
    state = initialize_search(
        root,
        data_directory,
        run_directory,
        node_directory,
        environment=environment,
        timeout_seconds=timeout_seconds,
    )
    attempted = 0

    for iteration in range(1, max_iterations + 1):
        remaining = wall_clock_seconds - (time.monotonic() - started)
        if remaining <= 0:
            stopped_reason = "wall_clock_cap"
            break
        outcome = run_search_iteration(
            iteration,
            state,
            run_directory,
            node_directory,
            root=root,
            data_directory=data_directory,
            environment=environment,
            timeout_seconds=min(timeout_seconds, remaining),
            client=client,
        )
        input_tokens += outcome.input_tokens
        output_tokens += outcome.output_tokens
        if outcome.stop_reason is not None:
            stopped_reason = outcome.stop_reason
            break
        attempted += 1
        if has_converged(state.scores):
            stopped_reason = "converged"
            break

    try:
        final_submission = finalize_submission(
            state.best,
            run_directory,
            root=root,
            data_directory=data_directory,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        final_submission = None
        stopped_reason = f"submission_failure: {error}"
    write_log(
        run_directory / "manifest.json",
        build_manifest(
            state,
            root=root,
            stopped_reason=stopped_reason,
            attempted=attempted,
            max_iterations=max_iterations,
            elapsed_seconds=time.monotonic() - started,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            final_submission=final_submission,
        ),
    )
    return AgentSummary(
        best_validation=state.best.metrics or state.baseline,
        iterations=attempted,
        manual_interventions=0,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        run_directory=run_directory,
        final_submission=final_submission,
    )
