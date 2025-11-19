from dataclasses import dataclass
from typing import List


@dataclass
class Product:
    id: str
    name: str
    category_path: List[str]
    specificity_score: float = 0.0  # Độ cụ thể của category
    token_overlap: int = 0          # Số token trùng với query