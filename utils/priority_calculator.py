import re
from typing import List, Dict
import numpy as np
from models.product import Product


def normalize_text(text: str) -> List[str]:
    """Chuẩn hóa và tách từ"""
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    return tokens


def calculate_category_specificity(category: str, all_categories: List[str]) -> float:
    """
    CÁCH 1: Tính độ cụ thể dựa trên Document Frequency (DF)
    
    Logic: Category càng hiếm trong toàn bộ dataset → càng cụ thể
    
    Ví dụ:
    - "Mỹ phẩm" xuất hiện trong 1000 products → DF cao → KHÔNG cụ thể
    - "Sữa rửa mặt tạo bọt" xuất hiện trong 10 products → DF thấp → CỤ THỂ
    
    Score = -log(DF) → DF càng thấp, score càng cao
    """
    category_tokens = set(normalize_text(category))
    
    if not category_tokens:
        return 0.0
    
    # Đếm số lần xuất hiện trong tất cả categories
    df_count = sum(
        1 for cat in all_categories 
        if category_tokens.issubset(set(normalize_text(cat)))
    )
    
    # Tránh log(0)
    df_count = max(df_count, 1)
    total = len(all_categories)
    
    # IDF-like score
    specificity = -np.log(df_count / total)
    
    return specificity


def calculate_token_overlap(query: str, category: str) -> int:
    """
    CÁCH 2: Đếm số token trùng khớp
    
    Query: "sữa rửa mặt tạo bọt"
    - "Sữa rửa mặt tạo bọt" → 5 tokens match → BEST
    - "Sữa rửa mặt dạng kem" → 3 tokens match → Medium
    - "La Roche-Posay" → 0 tokens match → Worst
    """
    query_tokens = set(normalize_text(query))
    category_tokens = set(normalize_text(category))
    
    overlap = len(query_tokens & category_tokens)
    return overlap


def calculate_category_length(category: str) -> int:
    """
    CÁCH 3: Độ dài category (số từ)
    
    Logic: Category dài thường cụ thể hơn
    
    - "Sữa rửa mặt tạo bọt cho da dầu mụn nhạy cảm" → 9 từ → CỤ THỂ
    - "Sữa rửa mặt" → 3 từ → Trung bình
    - "Mỹ phẩm" → 2 từ → Chung chung
    """
    tokens = normalize_text(category)
    return len(tokens)


def calculate_composite_priority(product: Product, query: str, 
                                 all_categories: List[str],
                                 alpha: float = 1.0,
                                 beta: float = 2.0,
                                 gamma: float = 0.5) -> float:
    """
    KẾT HỢP 3 metrics:
    
    Priority = alpha * Specificity + beta * TokenOverlap + gamma * Length
    
    Weights:
    - alpha = 1.0: Độ cụ thể quan trọng
    - beta = 2.0: Token overlap QUAN TRỌNG NHẤT (vì match với query)
    - gamma = 0.5: Độ dài ít quan trọng hơn
    
    Tự động ưu tiên categories cụ thể + match query tốt!
    """
    # 1. Specificity (IDF-based)
    specificity = calculate_category_specificity(product.name, all_categories)
    
    # 2. Token overlap với query
    overlap = calculate_token_overlap(query, product.name)
    
    # 3. Category length
    length = calculate_category_length(product.name)
    
    # Composite score
    priority = alpha * specificity + beta * overlap + gamma * length
    
    # Store in product
    product.specificity_score = specificity
    product.token_overlap = overlap
    
    return priority


def create_prioritized_products(categories: List[str], query: str, idx: int,
                                alpha: float = 1.0, 
                                beta: float = 2.0,
                                gamma: float = 0.5) -> List[Product]:
    """Tạo products và tính priority score tự động"""
    products = []
    
    for i, cat in enumerate(categories):
        product = Product(
            id=f"{idx}_{i}",
            name=cat,
            category_path=[cat]
        )
        
        # Tính composite priority
        priority = calculate_composite_priority(
            product, query, categories, alpha, beta, gamma
        )
        
        products.append((product, priority))
    
    # Sort theo priority giảm dần
    products.sort(key=lambda x: x[1], reverse=True)
    
    return [p for p, _ in products]