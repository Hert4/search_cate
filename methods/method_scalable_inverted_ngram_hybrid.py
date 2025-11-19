"""Scalable methods for handling 100K+ categories using Inverted Index and N-gram matching"""

from typing import List, Tuple, Optional, Dict, Set
import re
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
from models.product import Product
from models.cross_encoder import Qwen3CrossEncoder
from models.semantic_encoder import SemanticEncoder
from utils.text_normalization import normalize_text


@dataclass
class ScalableProduct:
    """Product class specifically for scalable methods"""
    id: str
    name: str
    category_path: List[str]


class InvertedIndex:
    """
    🔥 Inverted Index - giải pháp cho 100K+ categories
    
    Thay vì scan toàn bộ categories, chỉ lookup những categories chứa token từ query
    
    Structure:
    {
        "sữa": [cat_0, cat_5, cat_120, ...],
        "rửa": [cat_0, cat_5, cat_89, ...],
        "mặt": [cat_0, cat_5, cat_13, cat_89, ...],
        "iphone": [cat_456, cat_789, ...],
        ...
    }
    
    Query "sữa rửa mặt" → Chỉ check GIAO của 3 lists → ~10-100 candidates thay vì 100K!
    """
    
    def __init__(self):
        self.index: Dict[str, Set[int]] = defaultdict(set)  # token -> set of category indices
        self.categories: List[str] = []
        self.category_tokens: List[Set[str]] = []
        self.category_token_counts: Dict[str, int] = defaultdict(int)  # Global token frequency
        self.total_categories: int = 0
    
    def build(self, categories: List[str]):
        """
        Build inverted index - chỉ chạy 1 lần khi khởi tạo
        
        Time: O(N * M) với N = số categories, M = trung bình số tokens/category
        Space: O(T * K) với T = số unique tokens, K = trung bình số categories/token
        """
        print(f"Building inverted index for {len(categories)} categories...")
        
        self.categories = categories
        self.total_categories = len(categories)
        
        for idx, category in enumerate(categories):
            tokens = set(normalize_text(category))
            self.category_tokens.append(tokens)
            
            # Update inverted index
            for token in tokens:
                self.index[token].add(idx)
                self.category_token_counts[token] += 1
        
        print(f"✅ Indexed {len(self.index)} unique tokens")
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        Search using inverted index
        
        Time: O(Q * K + C * log(C)) với:
        - Q = số tokens trong query
        - K = trung bình số categories/token (~10-100)
        - C = số candidates (~10-100)
        
        → Thay vì O(100K), chỉ cần O(100) operations!
        """
        query_tokens = set(normalize_text(query))
        
        if not query_tokens:
            return []
        
        # Bước 1: Lấy candidates từ inverted index (GIAO của các token lists)
        candidate_indices = None
        
        for token in query_tokens:
            if token in self.index:
                if candidate_indices is None:
                    candidate_indices = self.index[token].copy()
                else:
                    # Intersection - chỉ giữ categories chứa TẤT CẢ tokens
                    candidate_indices &= self.index[token]
            else:
                # Token không tồn tại → không có kết quả
                return []
        
        if not candidate_indices:
            # Fallback: Union - lấy categories chứa BẤT KỲ token nào
            candidate_indices = set()
            for token in query_tokens:
                if token in self.index:
                    candidate_indices |= self.index[token]
        
        # Bước 2: Tính score cho candidates (chỉ ~10-100 categories, không phải 100K!)
        scored_candidates = []
        
        for idx in candidate_indices:
            category_tokens = self.category_tokens[idx]
            
            # Score 1: Token overlap
            overlap = len(query_tokens & category_tokens)
            overlap_ratio = overlap / len(query_tokens)
            
            # Score 2: IDF của tokens matched (ưu tiên rare tokens)
            idf_sum = 0.0
            for token in (query_tokens & category_tokens):
                df = self.category_token_counts[token]
                idf = np.log(self.total_categories / df)
                idf_sum += idf
            
            # Score 3: Category length penalty (ngắn hơn = tốt hơn)
            length_penalty = 1.0 / (1.0 + len(category_tokens))
            
            # Composite score
            score = overlap_ratio * 2.0 + idf_sum * 1.0 + length_penalty * 0.5
            
            scored_candidates.append((idx, score))
        
        # Bước 3: Sort và return top K
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates[:top_k]
    
    def get_category(self, idx: int) -> str:
        return self.categories[idx]


class NGramIndex:
    """
    🔥 N-gram Index - xử lý prefix/partial matching tốt hơn Prefix Tree
    
    Ví dụ:
    - Category: "Điện thoại iPhone 13"
    - N-grams: ["điện", "thoại", "iphone", "điện thoại", "thoại iphone", ...]
    
    Query: "iphone" → Match được "iphone" trong "Điện thoại iPhone 13"
    Query: "điện thoại iphone" → Match được bigram "điện thoại" + unigram "iphone"
    """
    
    def __init__(self, n: int = 3):
        self.n = n  # Max n-gram size
        self.index: Dict[str, Set[int]] = defaultdict(set)
        self.categories: List[str] = []
    
    def build(self, categories: List[str]):
        print(f"Building {self.n}-gram index for {len(categories)} categories...")
        
        self.categories = categories
        
        for idx, category in enumerate(categories):
            tokens = normalize_text(category)
            
            # Generate n-grams (1-gram to n-gram)
            for size in range(1, min(self.n + 1, len(tokens) + 1)):
                for i in range(len(tokens) - size + 1):
                    ngram = " ".join(tokens[i:i+size])
                    self.index[ngram].add(idx)
        
        print(f"✅ Indexed {len(self.index)} unique n-grams")
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Search using n-gram matching"""
        query_tokens = normalize_text(query)
        
        if not query_tokens:
            return []
        
        # Generate query n-grams
        query_ngrams = []
        for size in range(min(self.n, len(query_tokens)), 0, -1):  # Ưu tiên n-gram dài
            for i in range(len(query_tokens) - size + 1):
                ngram = " ".join(query_tokens[i:i+size])
                query_ngrams.append((ngram, size))
        
        # Find candidates
        candidate_scores = defaultdict(float)
        
        for ngram, size in query_ngrams:
            if ngram in self.index:
                weight = size  # N-gram dài = score cao hơn
                for idx in self.index[ngram]:
                    candidate_scores[idx] += weight
        
        # Sort and return
        scored_candidates = [(idx, score) for idx, score in candidate_scores.items()]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates[:top_k]
    
    def get_category(self, idx: int) -> str:
        return self.categories[idx]


def method_inverted_index(query: str, categories: List[str],
                          cross_encoder: Qwen3CrossEncoder,
                          semantic_encoder: Optional[SemanticEncoder],
                          idx: int) -> Tuple[str, float, str, Dict]:
    """
    🔥 METHOD: Inverted Index (SCALABLE!)
    
    Pipeline:
    1. Build inverted index (1 lần duy nhất)
    2. Query → Chỉ check ~10-100 candidates (không phải 100K!)
    3. Cross-encoder rerank
    
    Complexity:
    - Build: O(N * M) - chạy 1 lần
    - Query: O(Q * K + C * log C) với C ~ 10-100
    """
    # Build inverted index
    inv_index = InvertedIndex()
    inv_index.build(categories)
    
    # Search - chỉ scan ~10-100 candidates!
    candidates = inv_index.search(query, top_k=50)
    
    if not candidates:
        return categories[0], 0.0, "inverted_index", {"candidates": 0}
    
    # Cross-encoder rerank
    candidate_categories = [inv_index.get_category(candidate_idx) for candidate_idx, _ in candidates]
    pairs = [[query, cat] for cat in candidate_categories]
    scores = cross_encoder.predict(pairs)
    
    best_idx = np.argmax(scores)
    best_category = candidate_categories[best_idx]
    
    debug_info = {
        "candidates": len(candidates),
        "retrieval_score": candidates[best_idx][1]
    }
    
    return best_category, scores[best_idx], "inverted_index", debug_info


def method_ngram_index(query: str, categories: List[str],
                       cross_encoder: Qwen3CrossEncoder,
                       semantic_encoder: Optional[SemanticEncoder],
                       idx: int) -> Tuple[str, float, str, Dict]:
    """
    🔥 METHOD: N-gram Index (XỬ LÝ PARTIAL MATCHING TỐT HƠN PREFIX TREE!)
    
    Pipeline:
    1. Build n-gram index (1 lần duy nhất)
    2. Query → Match n-grams → ~10-100 candidates
    3. Cross-encoder rerank
    
    Ví dụ:
    - Query: "iphone 13"
    - Match được: "Điện thoại iPhone 13", "iPhone 13 Pro", "Phụ kiện iPhone",...
    """
    # Build n-gram index
    ngram_index = NGramIndex(n=3)
    ngram_index.build(categories)
    
    # Search
    candidates = ngram_index.search(query, top_k=50)
    
    if not candidates:
        return categories[0], 0.0, "ngram_index", {"candidates": 0}
    
    # Cross-encoder rerank
    candidate_categories = [ngram_index.get_category(candidate_idx) for candidate_idx, _ in candidates]
    pairs = [[query, cat] for cat in candidate_categories]
    scores = cross_encoder.predict(pairs)
    
    best_idx = np.argmax(scores)
    best_category = candidate_categories[best_idx]
    
    debug_info = {
        "candidates": len(candidates),
        "retrieval_score": candidates[best_idx][1]
    }
    
    return best_category, scores[best_idx], "ngram_index", debug_info


def method_hybrid(query: str, categories: List[str],
                  cross_encoder: Qwen3CrossEncoder,
                  semantic_encoder: SemanticEncoder,
                  idx: int) -> Tuple[str, float, str, Dict]:
    """
    🔥 METHOD: Hybrid (Inverted Index + Semantic)
    
    Pipeline:
    1. Inverted Index → ~20-30 candidates (lexical)
    2. Semantic search → ~20-30 candidates (semantic)
    3. Union → ~40-60 unique candidates
    4. Cross-encoder rerank
    """
    # Step 1: Inverted index retrieval
    inv_index = InvertedIndex()
    inv_index.build(categories)
    lexical_candidates = inv_index.search(query, top_k=30)
    
    # Step 2: Semantic retrieval (chỉ trên subset nếu có nhiều categories)
    # Để demo, ta semantic search trên tất cả - nhưng trong production, 
    # có thể pre-compute embeddings và dùng FAISS/ANN search
    query_emb = semantic_encoder.encode([query])[0]
    
    # Chỉ encode top 200 candidates từ lexical (nếu có) để tiết kiệm
    if len(lexical_candidates) > 0:
        top_lexical_indices = [idx for idx, _ in lexical_candidates[:200]]
        semantic_pool = [categories[i] for i in top_lexical_indices]
    else:
        semantic_pool = categories[:200]  # Fallback
    
    semantic_embs = semantic_encoder.encode(semantic_pool)
    similarities = [float(np.dot(query_emb, emb)) for emb in semantic_embs]
    semantic_candidates = sorted(
        zip(range(len(semantic_pool)), similarities),
        key=lambda x: x[1], reverse=True
    )[:30]
    
    # Step 3: Union
    all_candidate_indices = set()
    for idx, _ in lexical_candidates:
        all_candidate_indices.add(idx)
    
    if len(lexical_candidates) > 0:
        for local_idx, _ in semantic_candidates:
            global_idx = top_lexical_indices[local_idx]
            all_candidate_indices.add(global_idx)
    
    if not all_candidate_indices:
        return categories[0], 0.0, "hybrid", {"candidates": 0}
    
    # Step 4: Cross-encoder rerank
    candidate_categories = [categories[i] for i in all_candidate_indices]
    pairs = [[query, cat] for cat in candidate_categories]
    scores = cross_encoder.predict(pairs)
    
    best_idx = np.argmax(scores)
    best_category = candidate_categories[best_idx]
    
    debug_info = {
        "lexical_candidates": len(lexical_candidates),
        "semantic_candidates": len(semantic_candidates),
        "total_candidates": len(all_candidate_indices)
    }
    
    return best_category, scores[best_idx], "hybrid", debug_info