"""Pairwise-loss experiment built on the official FM baseline."""

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(
    os.environ.get("KUAI_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
)
STARTER_DIRECTORY = PROJECT_ROOT / "kuairand-starter-kit"
sys.path.insert(0, str(STARTER_DIRECTORY))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from baseline import FM  # noqa: E402
from data import encode  # noqa: E402
from evaluate import evaluate  # noqa: E402
from kuairand_dataset import load_dataset  # noqa: E402


def sample_pairs(
    users: list[str], labels: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one negative impression per positive from the same user."""
    groups: dict[str, list[list[int]]] = {}
    for index, (user, label) in enumerate(zip(users, labels, strict=True)):
        groups.setdefault(user, [[], []])[int(label)].append(index)

    positives: list[int] = []
    negatives: list[int] = []
    for negative_indices, positive_indices in groups.values():
        if positive_indices and negative_indices:
            positives.extend(positive_indices)
            negatives.extend(
                rng.choice(
                    negative_indices, len(positive_indices), replace=True
                ).tolist()
            )
    return np.asarray(positives, dtype=np.int32), np.asarray(negatives, dtype=np.int32)


def pairwise_step(model: FM, positive: np.ndarray, negative: np.ndarray) -> float:
    """Apply one Adam step for pairwise logistic loss."""
    batch_size = len(positive)
    positive_logits, positive_embeddings, positive_sums = model.logits(positive)
    negative_logits, negative_embeddings, negative_sums = model.logits(negative)
    margin = positive_logits - negative_logits
    gradient = (1.0 / (1.0 + np.exp(np.clip(margin, -30, 30))) / batch_size).astype(
        np.float32
    )

    positive_gradient = -gradient
    negative_gradient = gradient
    gradient_v = np.zeros_like(model.V)
    gradient_w = np.zeros_like(model.W)
    np.add.at(gradient_w, positive, positive_gradient[:, None])
    np.add.at(gradient_w, negative, negative_gradient[:, None])
    np.add.at(
        gradient_v,
        positive,
        positive_gradient[:, None, None]
        * (positive_sums[:, None, :] - positive_embeddings),
    )
    np.add.at(
        gradient_v,
        negative,
        negative_gradient[:, None, None]
        * (negative_sums[:, None, :] - negative_embeddings),
    )
    gradient_v += model.l2 * model.V
    gradient_w += model.l2 * model.W

    model.t += 1
    beta_one, beta_two, epsilon = 0.9, 0.999, 1e-8
    for parameters in (
        (model.V, gradient_v, model.mV, model.vV),
        (model.W, gradient_w, model.mW, model.vW),
    ):
        parameter, current_gradient, first_moment, second_moment = parameters
        first_moment *= beta_one
        first_moment += (1 - beta_one) * current_gradient
        second_moment *= beta_two
        second_moment += (1 - beta_two) * (current_gradient * current_gradient)
        corrected_first = first_moment / (1 - beta_one**model.t)
        corrected_second = second_moment / (1 - beta_two**model.t)
        parameter -= model.lr * corrected_first / (np.sqrt(corrected_second) + epsilon)

    return float(np.mean(np.logaddexp(0, -margin)))


def run_pairwise(
    splits: dict[str, list[tuple[object, ...]]],
    *,
    epochs: int = 40,
    batch_size: int = 8192,
    patience: int = 4,
    seed: int = 0,
    embedding_dimension: int = 16,
    output_directory: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Train pairwise FM and retain the best validation checkpoint.

    Test labels are never evaluated here. The autonomous controller only needs
    test predictions for the final submission file.
    """
    encoded, dimension = encode(splits)
    train_x, train_y, train_users = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    test_x, _, _ = encoded["test"]
    model = FM(dimension, k=embedding_dimension, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best_score = -1.0
    best_state: tuple[np.ndarray, np.ndarray, np.float32] | None = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        positive_indices, negative_indices = sample_pairs(train_users, train_y, rng)
        order = rng.permutation(len(positive_indices))
        started = time.time()
        losses = [
            pairwise_step(
                model,
                train_x[positive_indices[order[start : start + batch_size]]],
                train_x[negative_indices[order[start : start + batch_size]]],
            )
            for start in range(0, len(order), batch_size)
        ]
        validation = evaluate(valid_users, valid_y, model.predict(valid_x))
        print(
            f"  epoch {epoch:2d} | loss {np.mean(losses):.4f} | "
            f"valid GAUC {validation['GAUC']:.4f} "
            f"nDCG@5 {validation['nDCG@5']:.4f} "
            f"primary {validation['primary']:.4f} | {time.time() - started:.1f}s"
        )
        if validation["primary"] > best_score + 1e-5:
            best_score = validation["primary"]
            bad_epochs = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("pairwise training produced no checkpoint")
    model.V, model.W, model.b = best_state
    valid_scores = model.predict(valid_x)
    test_scores = model.predict(test_x)
    result = {"valid": evaluate(valid_users, valid_y, valid_scores)}
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        np.save(output_directory / "valid_scores.npy", valid_scores)
        np.save(output_directory / "test_scores.npy", test_scores)
        np.savez(
            output_directory / "model.npz",
            V=model.V,
            W=model.W,
            b=model.b,
        )
        (output_directory / "metrics.json").write_text(
            json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pairwise experiment against a supported KuaiRand split."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=str(STARTER_DIRECTORY / "KuaiRand-Pure" / "data"),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args(argv)
    splits = load_dataset(Path(args.data_dir))
    print({name: len(rows) for name, rows in splits.items()})
    output_directory = Path(args.output_dir) if args.output_dir else None
    result = run_pairwise(
        splits,
        epochs=args.epochs,
        seed=args.seed,
        embedding_dimension=args.k,
        output_directory=output_directory,
    )
    for split in result:
        metrics = result[split]
        print(
            f"{split:5s} GAUC {metrics['GAUC']:.4f} | "
            f"nDCG@5 {metrics['nDCG@5']:.4f} | primary {metrics['primary']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
