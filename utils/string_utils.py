import re


def remove_number_dot_space(text: str) -> str:
    """Remove leading numbering (e.g., '1. ') and surrounding quotes."""
    cleaned = re.sub(r"^\d+\.\s+", "", text)
    return cleaned.replace("'", "").replace('"', "").strip()
