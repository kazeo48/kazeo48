"""Generate training and testing datasets for math tutoring classification."""
from pathlib import Path

from ml.math_tutoring_classifier import DatasetConfig, generate_math_tutoring_dataset


def main() -> None:
    config = DatasetConfig(random_state=123)
    train_df = generate_math_tutoring_dataset(1000, config=config)
    test_df = generate_math_tutoring_dataset(100, config=DatasetConfig(random_state=999))

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved training data to {train_path.resolve()}")
    print(f"Saved testing data to {test_path.resolve()}")


if __name__ == "__main__":
    main()
