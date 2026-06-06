import re
from typing import Optional


def clean_text(text: Optional[str]) -> str:
    """Normalize a review while preserving Vietnamese context."""
    if text is None:
        return ""
    value = str(text).strip().lower()
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()
