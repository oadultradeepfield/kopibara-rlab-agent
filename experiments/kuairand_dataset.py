"""Load the standard KuaiRand splits across Pure and 1K layouts."""

import csv
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

Row = tuple[object, ...]
SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
FEEDBACK_COLUMNS = (
    "long_view",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
)
VIDEO_FEATURE_COLUMNS = (
    "video_type",
    "upload_type",
    "music_id",
    "music_type",
    "tag",
)
USER_FEATURE_COLUMNS = (
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
)


@dataclass(frozen=True, slots=True)
class RichInteraction:
    """One interaction with the auxiliary feedback used by deep experiments."""

    date: int
    user_id: str
    video_id: str
    author_id: str
    tab: str
    duration_ms: float
    hour: int
    time_ms: int
    is_rand: int
    feedback: tuple[int, ...]
    extra_features: tuple[str, ...]


def find_file(data_directory: Path, prefix: str) -> Path:
    """Find one dataset file without hard-coding its variant suffix."""
    matches = sorted(data_directory.glob(f"{prefix}*.csv"))
    if len(matches) != 1:
        raise ValueError(f"expected one {prefix} file, found {len(matches)}")
    return matches[0]


def read_authors(path: Path) -> dict[str, str]:
    """Read video-to-author IDs from a KuaiRand feature file."""
    with path.open(encoding="utf-8") as handle:
        return {
            row["video_id"]: row["author_id"]
            for row in csv.DictReader(handle)
            if row["video_id"] is not None and row["author_id"] is not None
        }


def read_feature_map(
    path: Path, key_column: str, feature_columns: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Read selected static categorical metadata."""
    with path.open(encoding="utf-8") as handle:
        return {
            row[key_column]: tuple(
                row.get(column) or "UNK" for column in feature_columns
            )
            for row in csv.DictReader(handle)
            if row[key_column] is not None
        }


def read_rows(path: Path, authors: dict[str, str]) -> list[Row]:
    """Read one standard log into the row shape expected by starter encode."""
    rows: list[Row] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            video_id = row["video_id"]
            rows.append(
                (
                    int(row["date"]),
                    row["user_id"],
                    video_id,
                    authors.get(video_id, "UNK"),
                    row["tab"],
                    float(row["duration_ms"]),
                    1 if row["long_view"] != "0" else 0,
                )
            )
    return rows


def read_rich_rows(
    path: Path,
    authors: dict[str, str],
    video_features: dict[str, tuple[str, ...]],
    user_features: dict[str, tuple[str, ...]],
) -> list[RichInteraction]:
    """Read interaction features and several labels for multi-task training."""
    rows: list[RichInteraction] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            video_id = row["video_id"] or "UNK"
            user_id = row["user_id"] or "UNK"
            rows.append(
                RichInteraction(
                    int(row["date"]),
                    user_id,
                    video_id,
                    authors.get(video_id, "UNK"),
                    row["tab"] or "UNK",
                    float(row["duration_ms"]),
                    int(row["hourmin"] or 0) // 100,
                    int(row["time_ms"] or 0),
                    int(row.get("is_rand") or 0),
                    tuple(
                        int((row.get(column) or "0") != "0")
                        for column in FEEDBACK_COLUMNS
                    ),
                    video_features.get(video_id, ("UNK",) * len(VIDEO_FEATURE_COLUMNS))
                    + user_features.get(user_id, ("UNK",) * len(USER_FEATURE_COLUMNS)),
                )
            )
    return rows


def mask_test_labels(rows: list[Row]) -> list[Row]:
    """Remove evaluation labels before rows reach candidate experiments."""
    return [row[:-1] + (0,) for row in rows]


def mask_test_feedback(rows: list[RichInteraction]) -> list[RichInteraction]:
    """Remove evaluation feedback before rows reach candidate experiments."""
    empty_feedback = (0,) * len(FEEDBACK_COLUMNS)
    return [replace(row, feedback=empty_feedback) for row in rows]


def load_dataset(data_directory: Path) -> dict[str, list[Row]]:
    """Load train, validation, and test rows from Pure or 1K data."""
    authors = read_authors(find_file(data_directory, "video_features_basic_"))
    rows = read_rows(find_file(data_directory, "log_standard_4_08_to_4_21_"), authors)
    rows.extend(
        read_rows(find_file(data_directory, "log_standard_4_22_to_5_08_"), authors)
    )
    splits = {
        name: [row for row in rows if bounds[0] <= row[0] <= bounds[1]]
        for name, bounds in SPLITS.items()
    }
    splits["test"] = mask_test_labels(splits["test"])
    return splits


def load_rich_dataset(data_directory: Path) -> dict[str, list[RichInteraction]]:
    """Load train, validation, and test rows with auxiliary feedback labels."""
    video_path = find_file(data_directory, "video_features_basic_")
    authors = read_authors(video_path)
    video_features = read_feature_map(video_path, "video_id", VIDEO_FEATURE_COLUMNS)
    user_features = read_feature_map(
        find_file(data_directory, "user_features_"), "user_id", USER_FEATURE_COLUMNS
    )
    rows = read_rich_rows(
        find_file(data_directory, "log_standard_4_08_to_4_21_"),
        authors,
        video_features,
        user_features,
    )
    rows.extend(
        read_rich_rows(
            find_file(data_directory, "log_standard_4_22_to_5_08_"),
            authors,
            video_features,
            user_features,
        )
    )
    splits = {
        name: [row for row in rows if bounds[0] <= row.date <= bounds[1]]
        for name, bounds in SPLITS.items()
    }
    splits["test"] = mask_test_feedback(splits["test"])
    return splits
