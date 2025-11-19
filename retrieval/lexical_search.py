from typing import List
from models.product import Product
from utils.text_normalization import normalize_text


def lexical_search(query: str, products: List[Product], top_n: int = 50) -> List[Product]:
    """Lexical search dựa trên token matching"""
    query_tokens = normalize_text(query)

    scores = []
    for product in products:
        all_text = product.name + " " + " ".join(product.category_path)
        product_tokens = normalize_text(all_text)

        matches = sum(1 for qt in query_tokens if qt in product_tokens)
        score = matches / len(query_tokens) if query_tokens else 0

        scores.append((product, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [p for p, s in scores[:top_n] if s > 0]