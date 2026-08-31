"""Grouped LightGBM LambdaRank experiment for within-user ranking."""

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
from kuairand_dataset import RichInteraction, load_rich_dataset  # noqa: E402

LEARNING_RATE = 0.05
ESTIMATOR_COUNT = 300
EARLY_STOPPING_ROUNDS = 25
NUM_LEAVES = 31
MIN_DATA_IN_LEAF = 50
SEED = 0


def build_features(row: RichInteraction) -> list[str]:
    """Build categorical ranking features from static metadata."""
    return [
        row.user_id,
        row.video_id,
        row.author_id,
        row.tab,
        str(int(row.duration_ms // 30_000)),
        str(row.hour),
        str(row.is_rand),
        *row.extra_features,
    ]


def encode_rows(
    splits: dict[str, list[RichInteraction]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, list[str]]], int]:
    """Encode categorical features with training-only vocabularies."""
    train_features = [build_features(row) for row in splits["train"]]
    field_count = len(train_features[0])
    vocabularies: list[dict[str, int]] = [{} for _ in range(field_count)]
    for row in train_features:
        for field, value in enumerate(row):
            vocabularies[field].setdefault(value, len(vocabularies[field]) + 1)

    encoded: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for split, rows in splits.items():
        values = np.zeros((len(rows), field_count), dtype=np.int32)
        labels = np.empty(len(rows), dtype=np.float32)
        users: list[str] = []
        for index, row in enumerate(rows):
            for field, value in enumerate(build_features(row)):
                values[index, field] = vocabularies[field].get(value, 0)
            labels[index] = row.feedback[0]
            users.append(row.user_id)
        encoded[split] = (values, labels, users)
    return encoded, field_count


def group_rows(
    fields: np.ndarray, labels: np.ndarray, users: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort rows by user and return the ordering and group sizes."""
    order = np.argsort(np.asarray(users), kind="stable")
    grouped_users = np.asarray(users, dtype=str)[order]
    _, starts, counts = np.unique(grouped_users, return_index=True, return_counts=True)
    return fields[order], labels[order], order, counts[np.argsort(starts)]


def run_lambdarank(
    splits: dict[str, list[RichInteraction]],
    *,
    seed: int = SEED,
    objective: str = "lambdarank",
    output_directory: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Train a LightGBM ranker and select by validation nDCG@5."""
    encoded, field_count = encode_rows(splits)
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
        categorical_feature=list(range(field_count)),
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        valid_fields,
        label=valid_labels,
        group=valid_groups,
        categorical_feature=list(range(field_count)),
        reference=train_set,
        free_raw_data=False,
    )
    ranker = lgb.train(
        {
            "objective": objective,
            "metric": "ndcg" if objective == "lambdarank" else "auc",
            "eval_at": [5],
            "learning_rate": LEARNING_RATE,
            "num_leaves": NUM_LEAVES,
            "min_data_in_leaf": MIN_DATA_IN_LEAF,
            "seed": seed,
            "verbosity": -1,
        },
        train_set,
        num_boost_round=ESTIMATOR_COUNT,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    valid_ordered_scores = ranker.predict(valid_fields)
    if objective == "lambdarank":
        valid_scores = np.empty(len(valid_ordered_scores), dtype=np.float64)
        valid_scores[valid[2]] = valid_ordered_scores
    else:
        valid_scores = valid_ordered_scores
    test_scores = ranker.predict(test_fields)
    result = {"valid": evaluate(encoded["valid"][2], encoded["valid"][1], valid_scores)}
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        np.save(output_directory / "valid_scores.npy", valid_scores)
        np.save(output_directory / "test_scores.npy", test_scores)
        ranker.save_model(str(output_directory / "model.txt"))
        (output_directory / "metrics.json").write_text(
            json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grouped LambdaRank experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(STARTER_DIRECTORY / "KuaiRand-Pure" / "data"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--objective", choices=("lambdarank", "binary"), default="lambdarank"
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    splits = load_rich_dataset(Path(args.data_dir))
    print({name: len(rows) for name, rows in splits.items()})
    result = run_lambdarank(
        splits,
        seed=args.seed,
        objective=args.objective,
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
