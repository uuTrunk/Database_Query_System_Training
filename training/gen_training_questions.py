import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'Database_Query_System_Agent'))
from pathlib import Path

import pandas as pd

import data_access.read_db
from llm_access import LLM, call_llm_test
from utils.paths import TRAINING_QUESTIONS_GRAPH_PATH
from utils.string_utils import remove_number_dot_space
from utils.write_csv import write_csv_from_list

pd.set_option("display.max_columns", None)


def fetch_data():
    """Fetch database content and schema metadata for prompt construction.

    Args:
        None.

    Returns:
        list: Three-item list ``[tables_data, foreign_keys, comments]`` sourced
        from ``data_access.read_db.get_data_from_db()``.
    """
    dict_data, key, comments = data_access.read_db.get_data_from_db()
    return [dict_data, key, comments]


def slice_dfs(df_dict, lines=5):
    """Slice each DataFrame to its top rows for lightweight prompt context.

    Args:
        df_dict: Mapping of table names to pandas DataFrame objects.
        lines: Maximum number of head rows kept per DataFrame.

    Returns:
        dict: Mapping with the same keys as ``df_dict`` and truncated DataFrames
        as values.
    """
    top_rows_dict = {}
    for key, df in df_dict.items():
        top_rows_dict[key] = df.head(min(lines, len(df)))
    return top_rows_dict


def _build_prompt(data, question_count: int, history_questions=None) -> str:
    """Build the LLM prompt used to generate new training questions.

    Args:
        data: Three-item structure containing table data, foreign keys, and
        comments.
        question_count: Number of questions requested from the model.
        history_questions: Optional iterable of previously generated questions
        to avoid duplicates.

    Returns:
        str: Prompt text containing schema context and generation requirements.
    """
    prompt = (
        "I am trying to test a natural language query system for intelligent, "
        "multi-table database queries, statistical computations, and chart generation.\n"
        "Please generate questions that often require multi-value outputs (for example, top x, bottom y).\n"
        "Try to cover diverse chart-friendly analysis requests.\n"
        "Here is the test database structure:\n"
        f"{slice_dfs(data[0])}\n"
        "Here are key constraints of the tables:\n"
        f"{data[1]}\n"
        f"Please give me {question_count} test-case questions in English.\n"
        "Questions should be split by a single newline and contain no extra explanations.\n"
        "Example:\n"
        "1. question text\n"
        "2. question text\n"
        "3. question text\n"
    )

    if history_questions:
        prompt += "\nThese questions are already asked, do not repeat them:\n"
        prompt += str(history_questions)
    return prompt


def gen_questions(file_name: str, batch: int = 50, lang: str = "en") -> bool:
    """Generate training questions and append them to a CSV file.

    Args:
        file_name: Output CSV path for generated questions.
        batch: Number of questions requested in one generation round.
        lang: Language code for generation. Only ``"en"`` is supported.

    Returns:
        bool: ``True`` when valid questions are generated and written;
        otherwise ``False``.

    Raises:
        ValueError: If unsupported language is requested.
    """
    if lang != "en":
        raise ValueError("Only English question generation is currently supported.")

    output_path = Path(file_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history_questions = None
    if output_path.exists():
        history_questions = output_path.read_text(encoding="utf-8").splitlines()

    data = fetch_data()
    prompt = _build_prompt(data, question_count=batch, history_questions=history_questions)

    llm = LLM.get_llm()
    answer = call_llm_test.call_llm(prompt, llm)

    try:
        raw_list = answer.split("\n")
        question_list = [remove_number_dot_space(item) for item in raw_list]
        question_list = [item for item in question_list if item]
        if not question_list:
            raise ValueError("LLM returned no valid questions.")

        write_csv_from_list(str(output_path), question_list)
        return True
    except Exception as exc:
        print(exc)
        return False


if __name__ == "__main__":
    gen_questions(str(TRAINING_QUESTIONS_GRAPH_PATH), batch=50)
