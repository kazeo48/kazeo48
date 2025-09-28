"""Train a classifier to predict math tutoring enrollment."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from ml.math_tutoring_classifier import build_model_pipeline


DATA_DIR = Path("data")
MODEL_DIR = Path("models")


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    X = df.drop(columns=["takes_math_tutoring"])
    y = df["takes_math_tutoring"]
    return X, y


def main() -> None:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Training or testing dataset not found. Run generate_datasets.py first."
        )

    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "math_tutoring_model.joblib"
    joblib.dump(pipeline, model_path)

    y_pred = pipeline.predict(X_test)

    print("Model saved to", model_path.resolve())
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
