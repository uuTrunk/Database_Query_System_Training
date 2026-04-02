from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"
APP_LOG_FILE = PROJECT_ROOT / "ask_ai.log"

OUTPUT_STORE_DIR = PROJECT_ROOT / "output_store"
ASK_GRAPH_OUTPUT_DIR = OUTPUT_STORE_DIR / "ask-graph"
DATA_LOG_DIR = OUTPUT_STORE_DIR / "data_log"
TMP_IMG_DIR = PROJECT_ROOT / "tmp_img"
LOGIN_PAGE_TEMPLATE_FILE = PROJECT_ROOT / "static" / "login" / "login_page.html"

TRAIN_LOG_DIR = PROJECT_ROOT / "train_logs"
SAVES_DIR = PROJECT_ROOT / "saves"
BEST_MODEL_PATH = SAVES_DIR / "model_best.pth"

TRAINING_QUESTIONS_GRAPH_PATH = PROJECT_ROOT / "gened_questions" / "training_questions_for_graph.csv"
ASK_GRAPH_LOG_CSV = DATA_LOG_DIR / "ask_graph_1.csv"


def ensure_directories(paths: Iterable[PathLike]) -> None:
    """Create directories if they do not exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def ensure_runtime_directories() -> None:
    """Create all runtime directories used by online inference and training scripts."""
    ensure_directories(
        [
            OUTPUT_STORE_DIR,
            ASK_GRAPH_OUTPUT_DIR,
            DATA_LOG_DIR,
            TMP_IMG_DIR,
            TRAIN_LOG_DIR,
            SAVES_DIR,
        ]
    )
