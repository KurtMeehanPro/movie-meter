# Movie Meter

An IMDB sentiment-analysis project based on Kaggle's **Bag of Words Meets Bags
of Popcorn** competition (`word2vec-nlp-tutorial`). The project progresses from
data exploration through classical and neural model experiments to a
self-contained Kaggle submission notebook.

## Final result

The selected model uses word-level TF-IDF features and logistic regression:

- Unigrams and bigrams
- Minimum document frequency of 2
- Maximum of 100,000 features
- Sublinear term frequency
- Logistic regression with `C=2`

| Evaluation | ROC-AUC |
| --- | ---: |
| Five-fold cross-validation | 0.96542 |
| Kaggle public leaderboard | 0.96518 |

The close agreement between cross-validation and the public leaderboard
indicates that the selected model generalized as expected.

## Project workflow

1. `notebooks/01_imdb_dataset_exploration.ipynb` examines dataset quality,
   duplicate reviews, review lengths, HTML cleanup, and deduplication.
2. `notebooks/02_classical_model_comparison.ipynb` compares TF-IDF and averaged
   Word2Vec features using logistic regression on the same stratified split.
3. `notebooks/03_pytorch_dnn.ipynb` evaluates a PyTorch neural network using
   TF-IDF inputs.
4. `notebooks/04_model_1_advanced_tuning.ipynb` performs leakage-safe,
   five-fold cross-validation over logistic-regression regularization values.
5. `notebooks/05_kaggle_submission.ipynb` trains the selected model on all
   cleaned labeled reviews and creates `submission.csv` from the unlabeled
   competition test set.

Notebook 5 is self-contained and supports both local and Kaggle execution. Its
metadata uses Kaggle's standard `python3` kernel. The experimental notebooks
use the locally registered `Python (kaggle_torch_py312)` kernel.

## Setup

Python 3.12 was used for the project. Create and activate a virtual environment,
then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Authenticate with Kaggle and accept the competition rules before downloading
the data.

## Dataset sanity check

Run:

```bash
python scripts/imdb_dataset_sanity.py
```

The script downloads missing competition files through KaggleHub into
`data/raw`, loads the compressed TSV files, and prints basic dataset details. A
successful check reports 25,000 labeled training rows and 25,000 unlabeled test
rows.

Downloaded data, generated submissions, Python caches, and local editor settings
are excluded from Git.

## Generate a submission

Open `notebooks/05_kaggle_submission.ipynb` and run all cells. Locally, the
notebook reads from `data/raw`; on Kaggle, it discovers the attached competition
files under `/kaggle/input`. It writes `submission.csv` to the current directory
locally or `/kaggle/working` on Kaggle.

The notebook validates that the submission contains the required `id` and
`sentiment` columns, preserves all 25,000 unique test IDs, contains no missing
values, and stores positive-sentiment probabilities between 0 and 1.
