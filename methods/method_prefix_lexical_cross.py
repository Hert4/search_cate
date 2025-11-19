from typing import List, Tuple, Optional, Dict
from models.product import Product
from models.cross_encoder import Qwen3CrossEncoder
from retrieval.prefix_tree import build_prefix_tree, find_matching_node
from retrieval.lexical_search import lexical_search
from utils.text_normalization import normalize_text
from utils.priority_calculator import create_prioritized_products


def method_adaptive_priority(query: str, categories: List[str],
                             cross_encoder: Qwen3CrossEncoder,
                             semantic_encoder: Optional,
                             idx: int,
                             use_semantic: bool = False,
                             alpha: float = 1.0,
                             beta: float = 2.0,
                             gamma: float = 0.5) -> Tuple[str, float, str, Dict]:
    """
    PHƯƠNG PHÁP MỚI: Adaptive Priority (Không cần hardcode keywords!)

    Pipeline:
    1. Tự động tính Priority cho tất cả categories (dựa trên Specificity + Token Overlap + Length)
    2. Prefix Tree matching
    3. Lấy top candidates theo priority
    4. Lexical search trong candidates
    5. Cross-encoder rerank
    """
    # Bước 1: Tạo products với AUTO PRIORITY
    prioritized_products = create_prioritized_products(
        categories, query, idx, alpha, beta, gamma
    )

    # Build prefix tree
    tree = build_prefix_tree(prioritized_products)

    # Bước 2: Prefix matching
    matched_node = find_matching_node(tree, query)

    if matched_node and matched_node.products:
        candidates = matched_node.products[:50]  # Lấy top 50 theo priority
        retrieval_method = "prefix"
    else:
        # Bước 3: Lexical search
        candidates = lexical_search(query, prioritized_products, top_n=50)
        retrieval_method = "lexical"

    if not candidates:
        return categories[0], 0.0, "no_candidates", {}

    # Bước 4: Cross-encoder rerank
    pairs = [[query, p.name] for p in candidates]
    scores = cross_encoder.predict(pairs)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_product = candidates[best_idx]

    # Debug info
    debug_info = {
        'specificity': best_product.specificity_score,
        'token_overlap': best_product.token_overlap,
        'category_length': len(normalize_text(best_product.name))
    }

    return best_product.name, scores[best_idx], retrieval_method, debug_info