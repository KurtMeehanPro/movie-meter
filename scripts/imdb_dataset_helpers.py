"""Helper functions for acquiring and inspecting the IMDB dataset."""

# standard library

# third-party
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
