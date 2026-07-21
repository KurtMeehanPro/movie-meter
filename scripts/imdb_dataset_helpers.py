"""Helper functions for acquiring and inspecting the IMDB dataset."""

# standard library
from collections.abc import Mapping

# third-party
from bs4 import BeautifulSoup
import kagglehub
import pandas as pd

# local
from imdb_dataset_config import COMPETITION, DATA_DIR, FILES, REVIEW_PREVIEW_LENGTH


def ensure_dataset_downloaded() -> None:
    """Download the configured competition with KaggleHub when files are absent."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if all(path.is_file() for path in FILES.values()):
        print(f"Kaggle files already present; using: {DATA_DIR}")
        return

    download_path = kagglehub.competition_download(
        COMPETITION,
        output_dir=str(DATA_DIR),
    )
    print(f"KaggleHub download path: {download_path}")


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and return fresh training and test DataFrames from disk."""
    ensure_dataset_downloaded()
    train = pd.read_csv(FILES["train"], sep="\t", compression="zip")
    test = pd.read_csv(FILES["test"], sep="\t", compression="zip")
    return train, test


def build_dataset_overview(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize shapes, missing values, and duplicates by dataset."""
    overview_rows = []
    for name, data in datasets.items():
        overview_rows.append(
            {
                "dataset": name,
                "rows": len(data),
                "columns": data.shape[1],
                "missing_values": int(data.isna().sum().sum()),
                "duplicate_ids": int(data["id"].duplicated().sum()),
                "duplicate_reviews": int(data["review"].duplicated().sum()),
            }
        )

    return pd.DataFrame(overview_rows).set_index("dataset")


def duplicate_review_examples(
    data: pd.DataFrame, number_of_groups: int = 2
) -> pd.DataFrame:
    """Return every row belonging to a few exact duplicate-review groups."""
    duplicate_rows = data.loc[data["review"].duplicated(keep=False)].copy()
    selected_reviews = duplicate_rows["review"].drop_duplicates().head(number_of_groups)
    examples = duplicate_rows.loc[duplicate_rows["review"].isin(selected_reviews)].copy()
    group_numbers = {review: number for number, review in enumerate(selected_reviews, start=1)}

    examples["duplicate_group"] = examples["review"].map(group_numbers)
    examples["review_preview"] = examples["review"].str.replace(r"\s+", " ", regex=True).str[:300]
    columns = ["duplicate_group", "id"]
    if "sentiment" in examples.columns:
        columns.append("sentiment")
    columns.append("review_preview")

    return examples.loc[:, columns].sort_values(["duplicate_group", "id"])


def print_summary(name: str, data: pd.DataFrame) -> None:
    """Print basic dataset details and two short review previews."""
    print(f"\n{name}")
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print("Sample reviews:")

    for index, row in data.head(2).iterrows():
        review = " ".join(str(row["review"]).split())
        suffix = "..." if len(review) > REVIEW_PREVIEW_LENGTH else ""
        label = f" sentiment={row['sentiment']}" if "sentiment" in data.columns else ""
        print(f"  {index}:{label} {review[:REVIEW_PREVIEW_LENGTH]}{suffix}")


def clean_reviews(data: pd.DataFrame) -> pd.DataFrame:
    """Return reviews without HTML or duplicate occurrences after the first."""
    if "review" not in data.columns:
        raise KeyError("Expected a 'review' column in the dataset.")

    cleaned = data.copy()
    cleaned["review"] = cleaned["review"].map(
        lambda review: BeautifulSoup(str(review), "html.parser").get_text(" ", strip=True)
    )
    cleaned["review"] = cleaned["review"].str.replace(r"\s+", " ", regex=True)

    duplicate_mask = cleaned["review"].duplicated(keep="first")
    return cleaned.loc[~duplicate_mask].reset_index(drop=True)


def build_cleanup_report(
    raw_datasets: Mapping[str, pd.DataFrame],
    cleaned_datasets: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize row removal and remaining cleanup signals by dataset."""
    if raw_datasets.keys() != cleaned_datasets.keys():
        raise ValueError("Raw and cleaned dataset names must match.")

    report_rows = []
    for name, raw_data in raw_datasets.items():
        cleaned_data = cleaned_datasets[name]
        report_rows.append(
            {
                "dataset": name,
                "raw_rows": len(raw_data),
                "clean_rows": len(cleaned_data),
                "removed_rows": len(raw_data) - len(cleaned_data),
                "remaining_duplicate_rows": int(
                    cleaned_data["review"].duplicated(keep=False).sum()
                ),
                "remaining_angle_bracket_patterns": int(
                    cleaned_data["review"].str.contains(r"<[^>]+>", regex=True).sum()
                ),
            }
        )

    return pd.DataFrame(report_rows).set_index("dataset")
