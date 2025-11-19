from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from models.product import Product
from utils.text_normalization import normalize_text


class BM25Retriever:
    def __init__(self, products: List[Product]):
        """
        Khởi tạo Index BM25. Việc này tốn thời gian nên chỉ làm 1 lần
        khi khởi động chương trình hoặc khi load data.
        """
        print(f"⏳ Đang khởi tạo BM25 Index cho {len(products)} sản phẩm...")
        self.products = products
        # Tokenize toàn bộ danh mục 1 lần duy nhất
        self.tokenized_corpus = [normalize_text(p.name) for p in products]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("✅ BM25 Index sẵn sàng!")

    def search(self, query: str, top_n: int = 50) -> List[Tuple[Product, float]]:
        """
        Tìm kiếm cực nhanh dựa trên Index đã xây
        Trả về list các tuple: (product, bm25_score)
        """
        tokenized_query = normalize_text(query)

        # Lấy điểm số cho tất cả documents
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Lấy Top N index có điểm cao nhất
        # Dùng argpartition nhanh hơn argsort khi N nhỏ so với tổng data
        if top_n > len(doc_scores):
            top_n = len(doc_scores)

        top_indexes = np.argpartition(doc_scores, -top_n)[-top_n:]

        # Sắp xếp lại top N theo thứ tự giảm dần (vì argpartition không sort)
        top_indexes = top_indexes[np.argsort(doc_scores[top_indexes])[::-1]]

        candidates = []
        for idx in top_indexes:
            score = doc_scores[idx]
            if score > 0: # Chỉ lấy kết quả có liên quan
                candidates.append((self.products[idx], score))

        return candidates


def fast_lexical_search(query: str, products: List[Product], top_n: int = 50):
    """
    BM25-based lexical search to replace the current lexical_search implementation
    NOTE: This function reinitializes the BM25 index each time and is only kept for compatibility.
    For better performance, use the BM25Retriever class instead.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("Please install rank_bm25 library: pip install rank-bm25")

    # Extract product names for BM25
    all_product_names = [p.name for p in products]

    # Tokenize the corpus
    tokenized_corpus = [normalize_text(doc) for doc in all_product_names]

    # Initialize BM25
    bm25 = BM25Okapi(tokenized_corpus)

    # Tokenize the query
    tokenized_query = normalize_text(query)

    # Calculate document scores using BM25
    doc_scores = bm25.get_scores(tokenized_query)

    # Get top N indices
    top_indexes = np.argsort(doc_scores)[::-1][:top_n]

    # Return candidates with score > 0
    candidates = []
    for idx in top_indexes:
        if doc_scores[idx] > 0:
            candidates.append(products[idx])

    return candidates