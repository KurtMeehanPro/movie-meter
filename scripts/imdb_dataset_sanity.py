"""Download and run basic sanity checks on the configured IMDB dataset."""
# standard library

# third-party
import kagglehub
import pandas as pd

# local
from imdb_dataset_config import COMPETITION, DATA_DIR, FILES, REVIEW_PREVIEW_LENGTH


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
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if all(path.is_file() for path in FILES.values()):
        download_path = DATA_DIR
        print(f"Kaggle files already present; using: {download_path}")
    else:
        download_path = kagglehub.competition_download(
            COMPETITION,
            output_dir=str(DATA_DIR),
        )
        print(f"KaggleHub download path: {download_path}")

    datasets = {}
    for name, path in FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Expected Kaggle file was not found: {path}")

        datasets[name] = pd.read_csv(path, sep="\t", compression="zip")

    print_summary("Labeled training data", datasets["train"])
    print_summary("Test data", datasets["test"])

    print("\nKaggleHub download and data access sanity check passed.")


if __name__ == "__main__":
    main()
