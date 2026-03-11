from pathlib import Path
import pandas as pd

from src.config import DATA_PATH, TARGET_COLUMN


def load_raw_data(data_path=DATA_PATH) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {path}")
    df = pd.read_csv(path)
    return df


def basic_data_summary(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "class_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


if __name__ == "__main__":
    df = load_raw_data()
    summary = basic_data_summary(df)

    print("Dataset loaded successfully.")
    print(f"Shape: {summary['shape']}")
    print(f"Columns: {summary['columns']}")
    print(f"Target column: {summary['target_column']}")
    print(f"Class distribution: {summary['class_distribution']}")
    print(f"Missing values: {summary['missing_values']}")