from typing import List, Tuple, Optional, Dict
import numpy as np
from models.product import Product
from models.cross_encoder import Qwen3CrossEncoder
from retrieval.bm25_search import BM25Retriever, fast_lexical_search
from utils.text_normalization import normalize_text
from utils.priority_calculator import create_prioritized_products


def method_optimized_pipeline(query: str, categories: List[str],
                              cross_encoder: Qwen3CrossEncoder,
                              semantic_encoder: Optional,
                              idx: int,
                              use_semantic: bool = False,
                              alpha: float = 1.0,
                              beta: float = 2.0,
                              gamma: float = 0.5) -> Tuple[str, float, str, Dict]:
    """
    PHƯƠNG PHÁP MỚI: Optimized Pipeline with BM25 + Cross-Encoder

    Uses optimized BM25 algorithm for retrieval, then Cross-Encoder for re-ranking

    Pipeline:
    1. Tự động tính Priority cho tất cả categories (dựa trên Specificity + Token Overlap + Length)
    2. Initialize BM25Retriever with prioritized products
    3. BM25 retrieval (top 50 candidates)
    4. Cross-encoder rerank
    """
    # Bước 1: Tạo products với AUTO PRIORITY
    prioritized_products = create_prioritized_products(
        categories, query, idx, alpha, beta, gamma
    )

    # Bước 2: Initialize BM25Retriever và search
    retriever = BM25Retriever(prioritized_products)
    raw_candidates = retriever.search(query, top_n=50)

    if not raw_candidates:
        # Fallback: trả về category đầu tiên nếu không có candidates
        return categories[0], 0.0, "no_candidates", {}

    # Chuyển đổi candidates để đưa vào Cross-Encoder
    candidate_products = [item[0] for item in raw_candidates]  # Lấy products từ (product, score) tuple

    # Bước 3: Cross-encoder rerank
    pairs = [[query, p.name] for p in candidate_products]
    scores = cross_encoder.predict(pairs)

    # Lấy candidate có điểm cross-encoder cao nhất
    best_idx = np.argmax(scores)
    best_product = candidate_products[best_idx]
    best_ce_score = scores[best_idx]

    # Lấy điểm BM25 của candidate tốt nhất
    best_bm25_score = raw_candidates[best_idx][1] if best_idx < len(raw_candidates) else 0.0

    # Debug info
    query_tokens = normalize_text(query)
    best_product_tokens = normalize_text(best_product.name)
    overlap = len(set(query_tokens) & set(best_product_tokens))

    debug_info = {
        'bm25_score': best_bm25_score,
        'ce_score': best_ce_score,
        'token_overlap': overlap,
        'candidates_found': len(raw_candidates)
    }

    return best_product.name, best_ce_score, "bm25_hybrid", debug_info


def method_adaptive_priority_bm25(query: str, categories: List[str],
                                  cross_encoder: Qwen3CrossEncoder,
                                  semantic_encoder: Optional,
                                  idx: int,
                                  use_semantic: bool = False,
                                  alpha: float = 1.0,
                                  beta: float = 2.0,
                                  gamma: float = 0.5) -> Tuple[str, float, str, Dict]:
    """
    PHƯƠNG PHÁP MỚI: Adaptive Priority with BM25 Lexical Search
    Uses BM25 algorithm for lexical search instead of prefix tree + manual lexical search

    Pipeline:
    1. Tự động tính Priority cho tất cả categories (dựa trên Specificity + Token Overlap + Length)
    2. BM25 lexical search để lấy candidates
    3. Cross-encoder rerank
    """
    # Bước 1: Tạo products với AUTO PRIORITY
    prioritized_products = create_prioritized_products(
        categories, query, idx, alpha, beta, gamma
    )

    # Bước 2: BM25 lexical search (using the less optimized function that reinitializes each time)
    candidates = fast_lexical_search(query, prioritized_products, top_n=50)
    retrieval_method = "bm25"

    if not candidates:
        # Fallback: trả về category đầu tiên nếu không có candidates
        return categories[0], 0.0, "no_candidates", {}

    # Bước 3: Cross-encoder rerank
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