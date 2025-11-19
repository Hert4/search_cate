import re
from typing import List
from collections import Counter


def normalize_text(text: str) -> List[str]:
    """Chuẩn hóa và tách từ"""
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    return tokens