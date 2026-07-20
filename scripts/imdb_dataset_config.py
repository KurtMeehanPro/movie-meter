"""Project-specific configuration for the IMDB dataset sanity check."""

# standard library
from pathlib import Path

# third-party

# local

# constants
COMPETITION = "word2vec-nlp-tutorial"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
REVIEW_PREVIEW_LENGTH = 160
FILES = {
    "train": DATA_DIR / "labeledTrainData.tsv.zip",
    "test": DATA_DIR / "testData.tsv.zip",
}
