from pathlib import Path

# Root of the project (automatically finds current folder where config.py lives)
PROJECT_PATH = Path(__file__).resolve().parent

# Data folders (relative to project root)
ANNOTATIONS_PATH = PROJECT_PATH / "data" / "preprocessed_annotations"
RAW_ANNOTATIONS_PATH = PROJECT_PATH / "data" / "raw_annotations"
DATA_PATH = PROJECT_PATH / "data" / "images"

# Where to save models/logs
MODELS_PATH = PROJECT_PATH / "pretrained_model"
LOGS_PATH = PROJECT_PATH / "logs"
