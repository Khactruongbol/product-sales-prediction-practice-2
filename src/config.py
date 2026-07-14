from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

RAW_DATA_PATH = RAW_DIR / "retail_store_inventory.csv"
SOURCE_METADATA_PATH = RAW_DIR / "source_metadata.json"
MODELING_DATA_PATH = PROCESSED_DIR / "modeling_data.csv"
TRAIN_DATA_PATH = PROCESSED_DIR / "train.csv"
VALIDATION_DATA_PATH = PROCESSED_DIR / "validation.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test.csv"
CLEANING_REPORT_PATH = PROCESSED_DIR / "cleaning_report.json"
DATA_DICTIONARY_PATH = PROCESSED_DIR / "data_dictionary.json"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
FEATURE_SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"
TRAINING_SUMMARY_PATH = METRICS_DIR / "training_summary.json"
BEST_MODEL_INFO_PATH = METRICS_DIR / "best_model.json"

DATASET_REF = "anirudhchauhan/retail-store-inventory-forecasting-dataset"
DATASET_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}"
DATASET_PAGE = f"https://www.kaggle.com/datasets/{DATASET_REF}"
RANDOM_SEED = 42

RAW_REQUIRED_COLUMNS = [
    "Date",
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Demand Forecast",
    "Price",
    "Discount",
    "Weather Condition",
    "Holiday/Promotion",
    "Competitor Pricing",
    "Seasonality",
]

CATEGORICAL_COLUMNS = [
    "store_id",
    "product_id",
    "category",
    "region",
    "weather_condition",
    "holiday_promotion",
    "seasonality",
]

NUMERIC_COLUMNS = [
    "inventory_level",
    "units_ordered",
    "price",
    "discount",
    "competitor_pricing",
    "year",
    "month",
    "quarter",
    "iso_week",
    "day_of_week",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "units_sold_lag_1",
    "units_sold_lag_7",
    "units_sold_lag_14",
    "units_sold_lag_28",
    "units_sold_rolling_mean_7",
    "units_sold_rolling_std_7",
    "units_sold_rolling_mean_28",
    "units_sold_rolling_std_28",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
TARGET_COLUMN = "units_sold"


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

