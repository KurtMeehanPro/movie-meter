# Movie Meter

An IMDB sentiment-analysis project based on Kaggle's **Bag of Words Meets Bags
of Popcorn** competition (`word2vec-nlp-tutorial`).

## Dataset sanity check

The initial workflow downloads the competition data with KaggleHub and verifies
that the labeled training and test datasets load cleanly. It does not train a
model yet.

The dataset workflow is organized into three files:

- `scripts/imdb_dataset_config.py` defines the competition, local data paths,
  expected files, and preview settings.
- `scripts/imdb_dataset_helpers.py` contains reusable download and summary
  helpers.
- `scripts/imdb_dataset_sanity.py` is the executable sanity-check entry point.

### Prerequisites

Use a Python 3.13 environment containing `kagglehub`, `pandas`,
`beautifulsoup4`, `scikit-learn`, `gensim`, and Jupyter. Authenticate with
Kaggle and accept the competition rules before the first download.

### Run

With the desired environment activated:

```bash
python scripts/imdb_dataset_sanity.py
```

The script will:

1. Create `data/raw` when needed.
2. Download the competition files through KaggleHub when they are absent.
3. Load the compressed training and test TSV files with pandas.
4. Print dataset shapes, columns, and truncated sample reviews.

A successful check currently reports:

```text
Labeled training data: (25000, 3)
Test data: (25000, 2)
KaggleHub download and data access sanity check passed.
```

Downloaded data is stored under `data/` and excluded from Git.

## Notebooks

- `notebooks/01_imdb_dataset_exploration.ipynb` examines dataset quality,
  exact duplicate reviews, review lengths, and the effects of HTML cleanup and
  deduplication.
- `notebooks/02_classical_model_comparison.ipynb` compares TF-IDF features and
  averaged Word2Vec features using logistic regression on the same stratified
  validation split.

Both notebooks use the registered `Python (kaggle_py313)` kernel. Notebook 02
is independently runnable and reloads fresh data through the shared helpers.
The first baseline run favored TF-IDF, with validation ROC-AUC of `0.9639`
versus `0.9325` for mean Word2Vec features.
