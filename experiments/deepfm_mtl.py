"""DeepFM multi-task ranker selected on validation only."""

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

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
from pairwise_fm import sample_pairs  # noqa: E402

EMBEDDING_DIMENSION = 16
HIDDEN_DIMENSION = 128
AUXILIARY_WEIGHT = 0.15
RANKING_WEIGHT = 0.5
POINTWISE_WEIGHT = 1.0
TASK_WEIGHTS = (1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.0)
EXPERT_COUNT = 2
TASK_EXPERT_COUNT = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 8192
DEFAULT_EPOCHS = 12
EARLY_STOP_PATIENCE = 3
SEED = 0


def build_features(rows: Sequence[RichInteraction]) -> list[list[str]]:
    """Build sparse categorical fields from raw interactions."""
    return [
        [
            row.user_id,
            row.video_id,
            row.author_id,
            row.tab,
            str(int(row.duration_ms // 30_000)),
            str(row.hour),
            str(row.is_rand),
            *row.extra_features,
        ]
        for row in rows
    ]


def encode_rows(
    splits: dict[str, list[RichInteraction]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, list[str]]], int, int]:
    """Encode categorical fields using training-only vocabularies."""
    train_features = build_features(splits["train"])
    field_count = len(train_features[0])
    vocabularies: list[dict[str, int]] = [{} for _ in range(field_count)]
    for row in train_features:
        for field, value in enumerate(row):
            vocabularies[field].setdefault(value, len(vocabularies[field]))
    unknown = [len(vocab) for vocab in vocabularies]
    dimensions = [len(vocab) + 1 for vocab in vocabularies]
    offsets = np.cumsum([0, *dimensions[:-1]], dtype=np.int32)

    encoded: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for split, rows in splits.items():
        features = build_features(rows)
        values = np.empty((len(rows), field_count), dtype=np.int64)
        labels = np.empty((len(rows), len(FEEDBACK_COLUMNS)), dtype=np.float32)
        users: list[str] = []
        for index, (feature_row, raw_row) in enumerate(
            zip(features, rows, strict=True)
        ):
            for field, value in enumerate(feature_row):
                values[index, field] = (
                    vocabularies[field].get(value, unknown[field]) + offsets[field]
                )
            labels[index] = raw_row.feedback
            users.append(raw_row.user_id)
        encoded[split] = (values, labels, users)
    return encoded, int(sum(dimensions)), field_count


class DeepFM(nn.Module):
    """Factorization-machine interactions plus a small shared neural tower."""

    def __init__(
        self, dimension: int, field_count: int, task_count: int, seed: int
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embeddings = nn.Embedding(dimension, EMBEDDING_DIMENSION)
        self.linear = nn.Embedding(dimension, 1)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.weight)
        self.tower = nn.Sequential(
            nn.Linear(field_count * EMBEDDING_DIMENSION, HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMENSION, EMBEDDING_DIMENSION),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(EMBEDDING_DIMENSION, 1) for _ in range(task_count)]
        )
        self.bias = nn.Parameter(torch.zeros(task_count))

    def forward(self, fields: Tensor) -> Tensor:
        """Return one logit per feedback task."""
        embeddings = self.embeddings(fields)
        summed = embeddings.sum(dim=1)
        factorization = 0.5 * (
            summed.square().sum(dim=1) - embeddings.square().sum(dim=(1, 2))
        )
        deep = self.tower(embeddings.flatten(start_dim=1))
        return torch.stack(
            [
                self.bias[task]
                + self.linear(fields).squeeze(-1).sum(dim=1)
                + factorization
                + head(deep).squeeze(-1)
                for task, head in enumerate(self.heads)
            ],
            dim=1,
        )


class PLE(nn.Module):
    """One-layer progressive extraction with shared and task experts."""

    def __init__(
        self, dimension: int, field_count: int, task_count: int, seed: int
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        input_dimension = field_count * EMBEDDING_DIMENSION + 1
        self.embeddings = nn.Embedding(dimension, EMBEDDING_DIMENSION)
        self.linear = nn.Embedding(dimension, 1)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.weight)
        self.shared_experts = nn.ModuleList(
            [self.make_expert(input_dimension) for _ in range(EXPERT_COUNT)]
        )
        self.task_experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        self.make_expert(input_dimension)
                        for _ in range(TASK_EXPERT_COUNT)
                    ]
                )
                for _ in range(task_count)
            ]
        )
        expert_count = EXPERT_COUNT + TASK_EXPERT_COUNT
        self.gates = nn.ModuleList(
            [nn.Linear(input_dimension, expert_count) for _ in range(task_count)]
        )
        self.heads = nn.ModuleList(
            [nn.Linear(EMBEDDING_DIMENSION, 1) for _ in range(task_count)]
        )
        self.bias = nn.Parameter(torch.zeros(task_count))

    @staticmethod
    def make_expert(input_dimension: int) -> nn.Sequential:
        """Build one expert tower."""
        return nn.Sequential(
            nn.Linear(input_dimension, HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMENSION, EMBEDDING_DIMENSION),
            nn.ReLU(),
        )

    def forward(self, fields: Tensor) -> Tensor:
        """Return one gated expert logit per feedback task."""
        embeddings = self.embeddings(fields)
        summed = embeddings.sum(dim=1)
        factorization = 0.5 * (
            summed.square().sum(dim=1) - embeddings.square().sum(dim=(1, 2))
        )
        base = torch.cat(
            (embeddings.flatten(start_dim=1), factorization[:, None]), dim=1
        )
        shared = [expert(base) for expert in self.shared_experts]
        predictions: list[Tensor] = []
        for task, task_experts in enumerate(self.task_experts):
            outputs = shared + [expert(base) for expert in task_experts]
            expert_values = torch.stack(outputs, dim=1)
            weights = torch.softmax(self.gates[task](base), dim=1)
            representation = (weights[:, :, None] * expert_values).sum(dim=1)
            predictions.append(
                self.bias[task]
                + self.linear(fields).squeeze(-1).sum(dim=1)
                + factorization
                + self.heads[task](representation).squeeze(-1)
            )
        return torch.stack(predictions, dim=1)


def make_pair_indices(
    users: list[str], labels: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample same-user long-view positive and negative indices."""
    return sample_pairs(users, labels[:, 0], rng)


def train_epoch(
    model: nn.Module,
    *,
    fields: np.ndarray,
    labels: np.ndarray,
    users: list[str],
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    auxiliary_weight: float,
    ranking_weight: float,
    pointwise_weight: float,
) -> float:
    """Train one epoch with auxiliary BCE and same-user pairwise loss."""
    positives, negatives = make_pair_indices(users, labels, rng)
    order = rng.permutation(len(positives))
    losses: list[float] = []
    model.train()
    for start in range(0, len(order), BATCH_SIZE):
        batch = order[start : start + BATCH_SIZE]
        positive = positives[batch]
        negative = negatives[batch]
        indices = np.concatenate((positive, negative))
        predictions = model(torch.from_numpy(fields[indices]))
        targets = torch.from_numpy(labels[indices])
        main_loss = nn.functional.binary_cross_entropy_with_logits(
            predictions[:, 0], targets[:, 0]
        )
        auxiliary_losses = nn.functional.binary_cross_entropy_with_logits(
            predictions[:, 1:], targets[:, 1:], reduction="none"
        ).mean(dim=0)
        auxiliary_weights = torch.tensor(TASK_WEIGHTS[1:], dtype=auxiliary_losses.dtype)
        auxiliary_loss = (
            auxiliary_losses * auxiliary_weights
        ).sum() / auxiliary_weights.sum()
        ranking_loss = nn.functional.softplus(
            predictions[len(batch) :, 0] - predictions[: len(batch), 0]
        ).mean()
        loss = (
            pointwise_weight * main_loss
            + auxiliary_weight * auxiliary_loss
            + ranking_weight * ranking_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def predict(model: nn.Module, fields: np.ndarray) -> np.ndarray:
    """Predict long-view logits without retaining gradients."""
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(fields), BATCH_SIZE * 4):
            logits = model(torch.from_numpy(fields[start : start + BATCH_SIZE * 4]))
            scores.append(logits[:, 0].numpy())
    return np.concatenate(scores)


def run_deepfm(
    splits: dict[str, list[RichInteraction]],
    *,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = SEED,
    auxiliary_weight: float = AUXILIARY_WEIGHT,
    ranking_weight: float = RANKING_WEIGHT,
    pointwise_weight: float = POINTWISE_WEIGHT,
    architecture: str = "deepfm",
    output_directory: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Train and select a DeepFM checkpoint using validation only."""
    encoded, dimension, field_count = encode_rows(splits)
    train_fields, train_labels, train_users = encoded["train"]
    valid_fields, valid_labels, valid_users = encoded["valid"]
    test_fields, _, _ = encoded["test"]
    model_type = PLE if architecture == "ple" else DeepFM
    model = model_type(dimension, field_count, len(FEEDBACK_COLUMNS), seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(seed)
    best_score = -1.0
    best_state: dict[str, Tensor] | None = None
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        loss = train_epoch(
            model,
            fields=train_fields,
            labels=train_labels,
            users=train_users,
            optimizer=optimizer,
            rng=rng,
            auxiliary_weight=auxiliary_weight,
            ranking_weight=ranking_weight,
            pointwise_weight=pointwise_weight,
        )
        valid_scores = predict(model, valid_fields)
        metrics = evaluate(valid_users, valid_labels[:, 0], valid_scores)
        print(
            f"  epoch {epoch:2d} | loss {loss:.4f} | "
            f"valid GAUC {metrics['GAUC']:.4f} "
            f"nDCG@5 {metrics['nDCG@5']:.4f} "
            f"primary {metrics['primary']:.4f} | {time.monotonic() - started:.1f}s"
        )
        if metrics["primary"] > best_score + 1e-5:
            best_score = metrics["primary"]
            bad_epochs = 0
            best_state = {
                name: parameter.detach().clone()
                for name, parameter in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"  early stop at epoch {epoch}")
                break
    if best_state is None:
        raise RuntimeError("DeepFM training produced no checkpoint")
    model.load_state_dict(best_state)
    valid_scores = predict(model, valid_fields)
    test_scores = predict(model, test_fields)
    result = {"valid": evaluate(valid_users, valid_labels[:, 0], valid_scores)}
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        np.save(output_directory / "valid_scores.npy", valid_scores)
        np.save(output_directory / "test_scores.npy", test_scores)
        torch.save(model.state_dict(), output_directory / "model.pt")
        (output_directory / "metrics.json").write_text(
            json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the DeepFM multi-task experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(STARTER_DIRECTORY / "KuaiRand-Pure" / "data"),
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--auxiliary-weight", type=float, default=AUXILIARY_WEIGHT)
    parser.add_argument("--ranking-weight", type=float, default=RANKING_WEIGHT)
    parser.add_argument("--pointwise-weight", type=float, default=POINTWISE_WEIGHT)
    parser.add_argument("--architecture", choices=("deepfm", "ple"), default="deepfm")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    splits = load_rich_dataset(Path(args.data_dir))
    print({name: len(rows) for name, rows in splits.items()})
    result = run_deepfm(
        splits,
        epochs=args.epochs,
        seed=args.seed,
        auxiliary_weight=args.auxiliary_weight,
        ranking_weight=args.ranking_weight,
        pointwise_weight=args.pointwise_weight,
        architecture=args.architecture,
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
