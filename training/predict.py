from pathlib import Path
from typing import Optional

import torch
from transformers import BertTokenizer

from training.model import BertRegressionModel
from utils.logger import setup_logger
from utils.paths import BEST_MODEL_PATH

logger = setup_logger(__name__)

_tokenizer: Optional[BertTokenizer] = None
_model: Optional[BertRegressionModel] = None
_device = torch.device("cpu")


def _load_predictor() -> None:
    """Lazy-load tokenizer and trained model for prediction.

    Args:
        None.

    Returns:
        None: Updates module-level tokenizer/model singletons when not loaded.

    Raises:
        FileNotFoundError: If the best model checkpoint is missing.
    """
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return

    model_path = Path(BEST_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Predict model not found: {model_path}. Please run python -m training.model first."
        )

    _tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
    _model = BertRegressionModel()
    _model.load_state_dict(torch.load(model_path, map_location=_device))
    _model.to(_device)
    _model.eval()
    logger.info("Predict model loaded from %s", model_path)


def predict(text: str) -> float:
    """Predict the success probability for one question request.

    Args:
        text: Input question text.

    Returns:
        float: Predicted success probability clamped to ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``text`` is empty or not a string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string")

    _load_predictor()
    assert _tokenizer is not None and _model is not None

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    input_ids = inputs["input_ids"].to(_device)
    attention_mask = inputs["attention_mask"].to(_device)

    with torch.no_grad():
        outputs = _model(input_ids, attention_mask)
        predicted = outputs.flatten().cpu().numpy()

    raw_score = float(predicted[0])
    # Success rate is expected in [0, 1], so clamp model drift.
    return max(0.0, min(raw_score, 1.0))


if __name__ == "__main__":
    texts = ["Your text here", "Another text here"]
    for sample in texts:
        print(f"Predicted: {predict(sample)}")

