"""Smoke test for loading the Kaggle IMDB sentiment data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
REVIEW_PREVIEW_LENGTH = 160


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


def main() -> None:
    labeled_train = pd.read_csv(
        DATA_DIR / "labeledTrainData.tsv.zip",
        sep="\t",
        compression="zip",
    )
    test = pd.read_csv(
        DATA_DIR / "testData.tsv.zip",
        sep="\t",
        compression="zip",
    )

    print_summary("Labeled training data", labeled_train)
    print_summary("Test data", test)


if __name__ == "__main__":
    main()
