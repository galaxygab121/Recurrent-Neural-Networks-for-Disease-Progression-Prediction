from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.config import DATA_PATH, TARGET_COLUMN

ZERO_AS_MISSING_COLUMNS = [
    "Blood Glucose",
    "Blood Pressure",
    "Skin Fold Thickness",
    "2-Hour Insulin",
    "BMI",
]


def load_data(data_path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def count_zero_values(df: pd.DataFrame, columns: list[str]) -> dict:
    return {col: int((df[col] == 0).sum()) for col in columns}


def replace_invalid_zeros_with_nan(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[col] = df[col].replace(0, np.nan)
    return df


def impute_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    df = df.copy()

    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    imputer = SimpleImputer(strategy=strategy)

    df[feature_columns] = imputer.fit_transform(df[feature_columns])
    return df


def preprocess_pima_data(data_path=DATA_PATH) -> tuple[pd.DataFrame, dict, dict]:
    df = load_data(data_path)

    zero_counts_before = count_zero_values(df, ZERO_AS_MISSING_COLUMNS)

    df = replace_invalid_zeros_with_nan(df, ZERO_AS_MISSING_COLUMNS)

    missing_after_zero_replacement = df.isna().sum().to_dict()

    df = impute_missing_values(df, strategy="median")

    return df, zero_counts_before, missing_after_zero_replacement


def save_processed_data(df: pd.DataFrame, output_path: str = "data/processed/Pima_Diabetes_clean.csv") -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    cleaned_df, zero_counts_before, missing_after_zero_replacement = preprocess_pima_data()

    print("Zero counts before cleaning:")
    print(zero_counts_before)

    print("\nMissing values after zero replacement:")
    print(missing_after_zero_replacement)

    print("\nMissing values after imputation:")
    print(cleaned_df.isna().sum().to_dict())

    save_processed_data(cleaned_df)
    print("\nSaved cleaned dataset to data/processed/Pima_Diabetes_clean.csv")