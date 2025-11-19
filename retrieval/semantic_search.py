from typing import List
from models.product import Product
from models.semantic_encoder import SemanticEncoder
import numpy as np


def semantic_search(query: str, products: List[Product],
                    semantic_encoder: SemanticEncoder, top_n: int = 50) -> List[Product]:
    """Semantic search dựa trên embeddings"""
    query_emb = semantic_encoder.encode([query])[0]

    texts = [p.name for p in products]
    product_embs = semantic_encoder.encode(texts)

    similarities = [float(np.dot(query_emb, emb)) for emb in product_embs]

    scored = list(zip(products, similarities))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [p for p, s in scored[:top_n]]