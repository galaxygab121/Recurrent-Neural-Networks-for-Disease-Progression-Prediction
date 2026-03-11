from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Pima_Diabetes.csv"
TARGET_COLUMN = "Class"

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "Times Pregnant",
    "Blood Glucose",
    "Blood Pressure",
    "Skin Fold Thickness",
    "2-Hour Insulin",
    "BMI",
    "Family History",
    "Age",
]

OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_MODELS_DIR = PROJECT_ROOT / "outputs" / "models"