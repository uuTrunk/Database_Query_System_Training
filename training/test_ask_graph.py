import sys
from pathlib import Path
import base64
import json
import os
import time

from config.get_config import config_data
from utils.get_time import get_time
from utils.http_client import HTTPClient, HTTPClientError, wait_for_server_ready
from utils.logger import setup_logger
from utils.paths import (
    APP_LOG_FILE,
    ASK_GRAPH_LOG_CSV,
    ASK_GRAPH_OUTPUT_DIR,
    TRAINING_QUESTIONS_GRAPH_PATH,
    ensure_runtime_directories,
)
from utils.read_csv import read_csv_to_list
from utils.write_csv import write_csv_from_list

logger = setup_logger(__name__, log_file=str(APP_LOG_FILE))


def _normalize_question(question) -> str:
    """Normalize question records into a clean text string.

    Args:
        question: Raw question entry from CSV reader output.

    Returns:
        str: Trimmed question text, or an empty string when unavailable.
    """
    if isinstance(question, str):
        return question.strip()
    if isinstance(question, (list, tuple)) and question:
        return str(question[0]).strip()
    return ""


def _decode_json(response: bytes) -> dict:
    """Decode UTF-8 JSON bytes into a dictionary.

    Args:
        response: Raw HTTP response body bytes.

    Returns:
        dict: Parsed JSON payload.

    Raises:
        ValueError: If response decoding or JSON parsing fails.
    """
    try:
        return json.loads(response.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON response: {exc}") from exc


def _save_image_if_exists(response_json: dict) -> None:
    """Save returned base64 image data to disk when response is successful.

    Args:
        response_json: Parsed response payload from the graph endpoint.

    Returns:
        None: Writes image file if ``file`` and ``image_data`` are present.
    """
    file_path = response_json.get("file")
    image_data_base64 = response_json.get("image_data")
    if not file_path or not image_data_base64:
        return

    file_name = os.path.basename(file_path)
    image_path = ASK_GRAPH_OUTPUT_DIR / file_name
    image_bytes = base64.b64decode(image_data_base64)
    image_path.write_bytes(image_bytes)
    logger.info("Image saved to %s", image_path)


def _safe_retry_pair(retries_used) -> tuple:
    """Normalize retry metadata to a two-value tuple.

    Args:
        retries_used: Response field that may store retry counters.

    Returns:
        tuple: ``(retry_0, retry_1)`` with fallback ``(0, 0)`` when input
        format is invalid.
    """
    if isinstance(retries_used, list) and len(retries_used) >= 2:
        return retries_used[0], retries_used[1]
    return 0, 0


def _append_result_log(request: dict, response_json: dict) -> None:
    """Write summary and detail log records for one graph request.

    Args:
        request: Original request payload.
        response_json: Parsed response payload.

    Returns:
        None: Appends CSV log rows and optional per-result detail logs.
    """
    retry_0, retry_1 = _safe_retry_pair(response_json.get("retries_used", [0, 0]))
    file_val = response_json.get("file", "")
    response_code = response_json.get("code", -1)

    write_csv_from_list(
        str(ASK_GRAPH_LOG_CSV),
        [
            get_time(),
            request["question"],
            request["retries"][0],
            request["retries"][1],
            "/",
            response_code,
            retry_0,
            retry_1,
            file_val,
            "/",
        ],
    )

    if file_val:
        detail_name = os.path.basename(file_val) + ".txt"
        detail_path = ASK_GRAPH_OUTPUT_DIR / detail_name
        write_csv_from_list(
            str(detail_path),
            [get_time(), request["question"], str(request), str(response_json), "/"],
        )


def _build_request(question_text: str) -> dict:
    """Build a standard graph request payload from question text.

    Args:
        question_text: User question to send to the graph endpoint.

    Returns:
        dict: Request payload with fixed concurrency and retry settings.
    """
    return {
        "question": question_text,
        "concurrent": [1, 1],
        "retries": [5, 5],
    }


if __name__ == "__main__":
    ensure_runtime_directories()

    host = "127.0.0.1"
    port = config_data.get("agent_port", 8000)
    if not wait_for_server_ready(host, port, logger=logger):
        raise SystemExit("Server is not reachable. Exiting.")

    if not TRAINING_QUESTIONS_GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Question file not found: {TRAINING_QUESTIONS_GRAPH_PATH}. Run python -m training.gen_training_questions first."
        )

    question_list = read_csv_to_list(str(TRAINING_QUESTIONS_GRAPH_PATH))
    logger.info("Loaded %s questions", len(question_list))

    client = HTTPClient(host=host, port=port, timeout=60, max_retries=1, backoff_seconds=1, logger=logger)

    for index, question in enumerate(question_list, start=1):
        question_text = _normalize_question(question)
        if not question_text:
            continue

        print(f"Processing question {index}/{len(question_list)}: {question_text[:60]}...")
        success = False

        for attempt in range(1, 4):
            request = _build_request(question_text)
            try:
                status_code, response_bytes = client.post_json(
                    "/ask/graph-steps",
                    request,
                    retries=0,
                )
            except HTTPClientError as exc:
                logger.error("Attempt %s failed: %s", attempt, exc)
                time.sleep(2)
                continue

            if status_code != 200:
                logger.warning("Attempt %s returned status=%s", attempt, status_code)
                time.sleep(1)
                continue

            try:
                response_json = _decode_json(response_bytes)
            except Exception as exc:
                logger.error("Attempt %s invalid JSON: %s", attempt, exc)
                time.sleep(1)
                continue

            try:
                if response_json.get("code") == 200:
                    _save_image_if_exists(response_json)
                _append_result_log(request, response_json)
                print("  Success")
                success = True
                break
            except Exception as exc:
                logger.exception("Attempt %s failed during output handling: %s", attempt, exc)
                time.sleep(1)

        if not success:
            print(f"  Give up on question: {question_text[:40]}...")
