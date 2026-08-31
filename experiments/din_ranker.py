"""Leakage-safe, compact target-aware history ranker."""

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
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
from kuairand_dataset import RichInteraction, load_rich_dataset  # noqa: E402
from pairwise_fm import sample_pairs  # noqa: E402

EMBEDDING_DIMENSION = 16
HIDDEN_DIMENSION = 128
HISTORY_SIZE = 20
BATCH_SIZE = 8192
LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 20
EARLY_STOP_PATIENCE = 4
SEED = 0
BASE_FIELD_COUNT = 5


def sort_rows(rows: Sequence[RichInteraction]) -> list[RichInteraction]:
    """Return interactions in chronological order."""
    return sorted(rows, key=lambda row: (row.date, row.time_ms))


def build_history_rows(
    rows: Sequence[RichInteraction],
    history_by_user: dict[str, deque[str]],
) -> list[list[str]]:
    """Build each row's history before adding that row to the user state."""
    histories: list[list[str]] = [[] for _ in rows]
    row_positions = {id(row): index for index, row in enumerate(rows)}
    for row in sort_rows(rows):
        index = row_positions[id(row)]
        histories[index] = list(history_by_user[row.user_id])
        history_by_user[row.user_id].append(row.video_id)
    return histories


def build_features(row: RichInteraction) -> list[str]:
    """Build non-video categorical fields for one interaction."""
    return [
        row.user_id,
        row.author_id,
        row.tab,
        str(int(row.duration_ms // 30_000)),
        str(row.hour),
    ]


def encode_rows(
    splits: dict[str, list[RichInteraction]],
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]],
    int,
    int,
]:
    """Encode fields and train-only chronological video histories."""
    train_rows = splits["train"]
    field_vocabularies: list[dict[str, int]] = [{} for _ in range(BASE_FIELD_COUNT)]
    video_vocabulary = {"UNK": 1}
    for row in train_rows:
        for field, value in enumerate(build_features(row)):
            field_vocabularies[field].setdefault(value, len(field_vocabularies[field]))
        video_vocabulary.setdefault(row.video_id, len(video_vocabulary) + 1)

    histories: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
    history_rows: dict[str, list[list[str]]] = {}
    history_rows["train"] = build_history_rows(train_rows, histories)
    for split in ("valid", "test"):
        # Test may use public validation history, but never hidden labels.
        history_rows[split] = build_history_rows(splits[split], histories.copy())

    field_offsets = np.cumsum(
        [0, *[len(vocabulary) + 1 for vocabulary in field_vocabularies[:-1]]],
        dtype=np.int64,
    )
    encoded: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]
    ] = {}
    for split, rows in splits.items():
        fields = np.empty((len(rows), BASE_FIELD_COUNT), dtype=np.int64)
        videos = np.empty(len(rows), dtype=np.int64)
        histories_array = np.zeros((len(rows), HISTORY_SIZE), dtype=np.int64)
        labels = np.empty(len(rows), dtype=np.float32)
        users: list[str] = []
        for index, row in enumerate(rows):
            for field, value in enumerate(build_features(row)):
                fields[index, field] = (
                    field_vocabularies[field].get(value, len(field_vocabularies[field]))
                    + field_offsets[field]
                )
            videos[index] = video_vocabulary.get(row.video_id, 1)
            history = history_rows[split][index][-HISTORY_SIZE:]
            for position, video_id in enumerate(history):
                histories_array[index, position] = video_vocabulary.get(video_id, 1)
            labels[index] = row.feedback[0]
            users.append(row.user_id)
        encoded[split] = (fields, videos, histories_array, labels, users)
    return (
        encoded,
        int(sum(len(vocabulary) + 1 for vocabulary in field_vocabularies)),
        len(video_vocabulary) + 1,
    )


class DINRanker(nn.Module):
    """Candidate-aware attention over a user's prior video IDs."""

    def __init__(self, field_dimension: int, video_dimension: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.fields = nn.Embedding(field_dimension, EMBEDDING_DIMENSION)
        self.videos = nn.Embedding(video_dimension, EMBEDDING_DIMENSION, padding_idx=0)
        nn.init.normal_(self.fields.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.videos.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.videos.weight[0])
        input_dimension = EMBEDDING_DIMENSION * 4
        self.tower = nn.Sequential(
            nn.Linear(input_dimension, HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMENSION, EMBEDDING_DIMENSION),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIMENSION, 1),
        )

    def forward(self, fields: Tensor, videos: Tensor, histories: Tensor) -> Tensor:
        """Score candidates using target-aware historical attention."""
        field_embedding = self.fields(fields).sum(dim=1)
        video_embedding = self.videos(videos)
        history_embedding = self.videos(histories)
        similarity = (history_embedding * video_embedding[:, None, :]).sum(dim=2)
        mask = histories != 0
        attention = torch.softmax(similarity.masked_fill(~mask, -1e9), dim=1)
        attention = attention * mask
        interest = (attention[:, :, None] * history_embedding).sum(dim=1)
        inputs = torch.cat(
            (field_embedding, video_embedding, interest, video_embedding * interest),
            dim=1,
        )
        return self.tower(inputs).squeeze(-1)


def train_epoch(
    model: DINRanker,
    *,
    fields: np.ndarray,
    videos: np.ndarray,
    histories: np.ndarray,
    labels: np.ndarray,
    users: list[str],
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
) -> float:
    """Train with same-user pairwise ranking loss."""
    positives, negatives = sample_pairs(users, labels, rng)
    order = rng.permutation(len(positives))
    model.train()
    losses: list[float] = []
    for start in range(0, len(order), BATCH_SIZE):
        batch = order[start : start + BATCH_SIZE]
        positive = positives[batch]
        negative = negatives[batch]
        indices = np.concatenate((positive, negative))
        scores = model(
            torch.from_numpy(fields[indices]),
            torch.from_numpy(videos[indices]),
            torch.from_numpy(histories[indices]),
        )
        margin = scores[: len(batch)] - scores[len(batch) :]
        loss = nn.functional.softplus(-margin).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def predict(
    model: DINRanker,
    fields: np.ndarray,
    videos: np.ndarray,
    histories: np.ndarray,
) -> np.ndarray:
    """Predict scores without retaining gradients."""
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(fields), BATCH_SIZE * 4):
            scores.append(
                model(
                    torch.from_numpy(fields[start : start + BATCH_SIZE * 4]),
                    torch.from_numpy(videos[start : start + BATCH_SIZE * 4]),
                    torch.from_numpy(histories[start : start + BATCH_SIZE * 4]),
                ).numpy()
            )
    return np.concatenate(scores)


def run_din(
    splits: dict[str, list[RichInteraction]],
    *,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = SEED,
    output_directory: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Train and select a history ranker with validation-only model selection."""
    encoded, field_dimension, video_dimension = encode_rows(splits)
    train = encoded["train"]
    valid = encoded["valid"]
    test = encoded["test"]
    model = DINRanker(field_dimension, video_dimension, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(seed)
    best_score = -1.0
    best_state: dict[str, Tensor] | None = None
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        loss = train_epoch(
            model,
            fields=train[0],
            videos=train[1],
            histories=train[2],
            labels=train[3],
            users=train[4],
            optimizer=optimizer,
            rng=rng,
        )
        valid_scores = predict(model, valid[0], valid[1], valid[2])
        metrics = evaluate(valid[4], valid[3], valid_scores)
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
        raise RuntimeError("DIN training produced no checkpoint")
    model.load_state_dict(best_state)
    valid_scores = predict(model, valid[0], valid[1], valid[2])
    test_scores = predict(model, test[0], test[1], test[2])
    result = {"valid": evaluate(valid[4], valid[3], valid_scores)}
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
    """Run the chronological history experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(STARTER_DIRECTORY / "KuaiRand-Pure" / "data"),
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    splits = load_rich_dataset(Path(args.data_dir))
    print({name: len(rows) for name, rows in splits.items()})
    result = run_din(
        splits,
        epochs=args.epochs,
        seed=args.seed,
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
