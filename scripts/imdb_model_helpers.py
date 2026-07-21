"""Reusable helpers for training and comparing IMDB sentiment models."""

# standard library
from collections.abc import Sequence

# third-party
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# local


def split_labeled_reviews(
    data: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create a reproducible stratified train-validation split."""
    return train_test_split(
        data["review"],
        data["sentiment"],
        test_size=validation_size,
        random_state=random_state,
        stratify=data["sentiment"],
    )


def average_word_vectors(
    tokenized_documents: Sequence[list[str]],
    word_vectors,
) -> np.ndarray:
    """Represent each tokenized document by its mean in-vocabulary word vector."""
    document_vectors = []
    for tokens in tokenized_documents:
        vectors = [word_vectors[token] for token in tokens if token in word_vectors]
        if vectors:
            document_vectors.append(np.mean(vectors, axis=0))
        else:
            document_vectors.append(np.zeros(word_vectors.vector_size, dtype=np.float32))

    return np.asarray(document_vectors)


def evaluate_classifier(
    name: str,
    classifier,
    features,
    labels: pd.Series,
    fit_seconds: float,
    feature_count: int,
) -> dict[str, float | int | str]:
    """Return common binary-classification metrics for a fitted classifier."""
    predictions = classifier.predict(features)
    probabilities = classifier.predict_proba(features)[:, 1]
    return {
        "model": name,
        "roc_auc": roc_auc_score(labels, probabilities),
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "features": feature_count,
        "fit_seconds": fit_seconds,
    }
