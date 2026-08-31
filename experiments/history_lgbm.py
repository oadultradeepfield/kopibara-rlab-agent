"""Leakage-safe historical target features with a small LightGBM model."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import lightgbm as lgb
import numpy as np

PROJECT_ROOT = Path(
    os.environ.get("KUAI_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
)
STARTER_DIRECTORY = PROJECT_ROOT / "kuairand-starter-kit"
sys.path.insert(0, str(STARTER_DIRECTORY))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from evaluate import evaluate  # noqa: E402
from kuairand_dataset import (  # noqa: E402
    FEEDBACK_COLUMNS,
    RichInteraction,
    load_rich_dataset,
)

LEARNING_RATE = 0.05
ESTIMATOR_COUNT = 400
EARLY_STOPPING_ROUNDS = 30
NUM_LEAVES = 31
MIN_DATA_IN_LEAF = 50
SEED = 0
HISTORY_SCOPE_COUNT = 4


def build_categorical_features(row: RichInteraction) -> list[str]:
    """Build categorical context features known before feedback is observed."""
    return [
        row.user_id,
        row.video_id,
        row.author_id,
        row.tab,
        str(int(row.duration_ms // 30_000)),
        str(row.hour),
        str(row.is_rand),
        str(row.date),
        str((row.time_ms // (15 * 60 * 1000)) % 96),
        str(row.date % 7),
        *row.extra_features,
    ]


def build_context_features(row: RichInteraction) -> list[float]:
    """Build compact continuous context features."""
    minute_of_day = (row.time_ms // 60_000) % (24 * 60)
    return [
        float(np.log1p(row.duration_ms)),
        float(row.duration_ms / 1000.0),
        float(row.hour),
        float(minute_of_day / (24 * 60)),
        float(row.date - 20220408),
        float(row.is_rand),
    ]


def scope_keys(row: RichInteraction) -> tuple[str, ...]:
    """Return keys for user, item, author, and user-item histories."""
    return (
        row.user_id,
        row.video_id,
        row.author_id,
        f"{row.user_id}\x00{row.video_id}",
    )


def history_features(
    states: list[dict[str, np.ndarray]],
    last_seen: list[dict[str, int]],
    row: RichInteraction,
    feedback_indices: tuple[int, ...],
) -> list[float]:
    """Summarize only feedback from earlier interactions."""
    features: list[float] = []
    for scope, key in enumerate(scope_keys(row)):
        stats = states[scope].get(key)
        if stats is None:
            features.extend([0.0] * (2 + 2 * len(feedback_indices)))
        else:
            count = stats[0]
            sums = stats[1:]
            rates = sums / count
            features.extend(
                [
                    float(np.log1p(count)),
                    float(np.log1p(last_seen[scope][key])),
                    *rates.tolist(),
                    *np.log1p(sums).tolist(),
                ]
            )
        previous_time = last_seen[scope].get(key)
        hours_since = (
            30.0
            if previous_time is None
            else max(row.time_ms - previous_time, 0) / (60 * 60 * 1000)
        )
        if previous_time is None:
            features.append(30.0)
        else:
            features.append(float(np.log1p(hours_since)))
    return features


def update_history(
    states: list[dict[str, np.ndarray]],
    last_seen: list[dict[str, int]],
    row: RichInteraction,
    feedback_indices: tuple[int, ...],
) -> None:
    """Add one observed train or public-validation interaction to history."""
    feedback = np.asarray(
        [row.feedback[index] for index in feedback_indices], dtype=np.float64
    )
    for scope, key in enumerate(scope_keys(row)):
        stats = states[scope].setdefault(
            key, np.zeros(1 + len(feedback_indices), dtype=np.float64)
        )
        stats[0] += 1.0
        stats[1:] += feedback
        last_seen[scope][key] = row.time_ms


def encode_features(
    splits: dict[str, list[RichInteraction]],
    feedback_indices: tuple[int, ...],
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    int,
]:
    """Create base and history features with split-safe label updates."""
    train_rows = splits["train"]
    categorical_rows = [build_categorical_features(row) for row in train_rows]
    field_count = len(categorical_rows[0])
    vocabularies: list[dict[str, int]] = [{} for _ in range(field_count)]
    for feature_row in categorical_rows:
        for field, value in enumerate(feature_row):
            vocabularies[field].setdefault(value, len(vocabularies[field]) + 1)
    unknown = [len(vocab) + 1 for vocab in vocabularies]

    all_rows: list[tuple[str, int, RichInteraction]] = [
        (split, index, row)
        for split, rows in splits.items()
        for index, row in enumerate(rows)
    ]
    order = sorted(range(len(all_rows)), key=lambda index: all_rows[index][2].time_ms)
    encoded: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    values_by_split: dict[str, np.ndarray] = {}
    labels_by_split: dict[str, np.ndarray] = {}
    users_by_split: dict[str, list[str]] = {}
    history_by_split: dict[str, list[list[float]]] = {
        split: [[] for _ in rows] for split, rows in splits.items()
    }
    for split, rows in splits.items():
        values_by_split[split] = np.empty((len(rows), field_count), dtype=np.int32)
        labels_by_split[split] = np.empty(len(rows), dtype=np.float32)
        users_by_split[split] = [row.user_id for row in rows]
        for index, row in enumerate(rows):
            for field, value in enumerate(build_categorical_features(row)):
                values_by_split[split][index, field] = vocabularies[field].get(
                    value, unknown[field]
                )
            labels_by_split[split][index] = row.feedback[0]

    states: list[dict[str, np.ndarray]] = [{} for _ in range(HISTORY_SCOPE_COUNT)]
    last_seen: list[dict[str, int]] = [{} for _ in range(HISTORY_SCOPE_COUNT)]
    for position in order:
        split, index, row = all_rows[position]
        history_by_split[split][index] = [
            *build_context_features(row),
            *history_features(states, last_seen, row, feedback_indices),
        ]
        if split in {"train", "valid"}:
            update_history(states, last_seen, row, feedback_indices)

    for split, _rows in splits.items():
        encoded[split] = (
            np.hstack(
                (
                    values_by_split[split],
                    np.asarray(history_by_split[split], dtype=np.float32),
                )
            ),
            labels_by_split[split],
            users_by_split[split],
        )
    return encoded, field_count


def group_rows(
    fields: np.ndarray, labels: np.ndarray, users: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort rows by user and return the ordering and group sizes."""
    order = np.argsort(np.asarray(users), kind="stable")
    grouped_users = np.asarray(users, dtype=str)[order]
    _, starts, counts = np.unique(grouped_users, return_index=True, return_counts=True)
    return fields[order], labels[order], order, counts[np.argsort(starts)]


def run_history_model(
    splits: dict[str, list[RichInteraction]],
    *,
    seed: int = SEED,
    objective: str = "lambdarank",
    feedback_indices: tuple[int, ...] = tuple(range(len(FEEDBACK_COLUMNS))),
    learning_rate: float = LEARNING_RATE,
    num_leaves: int = NUM_LEAVES,
    min_data_in_leaf: int = MIN_DATA_IN_LEAF,
    estimator_count: int = ESTIMATOR_COUNT,
    output_directory: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Train LightGBM on context and leakage-safe historical features."""
    encoded, categorical_count = encode_features(splits, feedback_indices)
    if objective == "lambdarank":
        train = group_rows(*encoded["train"][:2], encoded["train"][2])
        valid = group_rows(*encoded["valid"][:2], encoded["valid"][2])
        train_fields, train_labels = train[0], train[1]
        valid_fields, valid_labels = valid[0], valid[1]
        train_groups, valid_groups = train[3], valid[3]
    else:
        train_fields, train_labels, _ = encoded["train"]
        valid_fields, valid_labels, _ = encoded["valid"]
        train_groups = valid_groups = None
    test_fields, _, _ = encoded["test"]
    train_set = lgb.Dataset(
        train_fields,
        label=train_labels,
        group=train_groups,
        categorical_feature=list(range(categorical_count)),
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        valid_fields,
        label=valid_labels,
        group=valid_groups,
        categorical_feature=list(range(categorical_count)),
        reference=train_set,
        free_raw_data=False,
    )
    model = lgb.train(
        {
            "objective": objective,
            "metric": "ndcg" if objective == "lambdarank" else "auc",
            "eval_at": [5],
            "lambdarank_truncation_level": 5,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "seed": seed,
            "verbosity": -1,
        },
        train_set,
        num_boost_round=estimator_count,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    valid_ordered_scores = model.predict(valid_fields)
    valid_scores = np.empty(len(valid_ordered_scores), dtype=np.float64)
    if objective == "lambdarank":
        valid_scores[valid[2]] = valid_ordered_scores
    else:
        valid_scores = valid_ordered_scores
    test_scores = model.predict(test_fields)
    result = {"valid": evaluate(encoded["valid"][2], encoded["valid"][1], valid_scores)}
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        np.save(output_directory / "valid_scores.npy", valid_scores)
        np.save(output_directory / "test_scores.npy", test_scores)
        model.save_model(str(output_directory / "model.txt"))
        (output_directory / "metrics.json").write_text(
            json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the historical-feature experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(STARTER_DIRECTORY / "KuaiRand-Pure" / "data"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--objective", choices=("lambdarank", "binary"), default="lambdarank"
    )
    parser.add_argument(
        "--feedbacks",
        default="long_view",
        help="comma-separated feedback names for history features, or all",
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-leaves", type=int, default=NUM_LEAVES)
    parser.add_argument("--min-data-in-leaf", type=int, default=MIN_DATA_IN_LEAF)
    parser.add_argument("--estimators", type=int, default=ESTIMATOR_COUNT)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    feedback_indices = (
        tuple(range(len(FEEDBACK_COLUMNS)))
        if args.feedbacks == "all"
        else tuple(FEEDBACK_COLUMNS.index(name) for name in args.feedbacks.split(","))
    )
    splits = load_rich_dataset(Path(args.data_dir))
    print({name: len(rows) for name, rows in splits.items()})
    result = run_history_model(
        splits,
        seed=args.seed,
        objective=args.objective,
        feedback_indices=feedback_indices,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        estimator_count=args.estimators,
        output_directory=Path(args.output_dir) if args.output_dir else None,
    )
    metrics = result["valid"]
    print(
        f"valid GAUC {metrics['GAUC']:.4f} | "
        f"nDCG@5 {metrics['nDCG@5']:.4f} | primary {metrics['primary']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
