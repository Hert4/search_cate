import re
from typing import List


def parse_categories(category_str: str) -> List[str]:
    """Parse danh mục từ string"""
    category_str = category_str.strip()
    if category_str.startswith('[') and category_str.endswith(']'):
        category_str = category_str[1:-1]
    categories = [cat.strip() for cat in category_str.split(',')]
    return categories