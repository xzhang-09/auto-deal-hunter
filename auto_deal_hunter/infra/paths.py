from pathlib import Path

# paths.py lives at <repo>/auto_deal_hunter/infra/paths.py, so the repo root
# (which holds data/) is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MEMORY_PATH = DATA_DIR / "memory.json"
DEFAULT_DEALS_DB_PATH = DATA_DIR / "deals.sqlite"
DEFAULT_VECTORSTORE_PATH = DATA_DIR / "products_vectorstore"
DEFAULT_EVAL_HOLDOUT_PATH = DATA_DIR / "eval_holdout.json"


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    DEFAULT_VECTORSTORE_PATH.mkdir(exist_ok=True)
