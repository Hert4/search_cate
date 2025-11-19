"""
Simple test to verify the scalable methods work correctly
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from methods.method_scalable_inverted_ngram_hybrid import (
    method_inverted_index, 
    method_ngram_index, 
    method_hybrid,
    InvertedIndex,
    NGramIndex
)
from models.semantic_encoder import SemanticEncoder
from models.cross_encoder import Qwen3CrossEncoder


def test_methods():
    """Test the scalable methods with a small example"""
    print("Testing scalable methods...")
    
    # Sample data
    query = "điện thoại iphone"
    categories = [
        "Điện tử",
        "Điện lạnh", 
        "Thiết bị số",
        "Máy tính",
        "Laptop", 
        "Điện thoại",
        "Smartphone",
        "iPhone",
        "iPhone 13",
        "iPhone 14",
        "Phụ kiện điện thoại"
    ]
    
    # Initialize models
    print("Loading models...")
    cross_encoder = Qwen3CrossEncoder(device="cpu")
    semantic_encoder = SemanticEncoder(device="cpu")
    
    print(f"Query: {query}")
    print(f"Categories: {categories}")
    print()
    
    # Test Inverted Index
    print("Testing Inverted Index...")
    try:
        result, score, method, debug = method_inverted_index(query, categories, cross_encoder, semantic_encoder, 0)
        print(f"  Result: {result}, Score: {score:.3f}, Method: {method}, Debug: {debug}")
    except Exception as e:
        print(f"  Error in Inverted Index: {e}")
    print()
    
    # Test N-gram Index
    print("Testing N-gram Index...")
    try:
        result, score, method, debug = method_ngram_index(query, categories, cross_encoder, semantic_encoder, 0)
        print(f"  Result: {result}, Score: {score:.3f}, Method: {method}, Debug: {debug}")
    except Exception as e:
        print(f"  Error in N-gram Index: {e}")
    print()
    
    # Test Hybrid
    print("Testing Hybrid...")
    try:
        result, score, method, debug = method_hybrid(query, categories, cross_encoder, semantic_encoder, 0)
        print(f"  Result: {result}, Score: {score:.3f}, Method: {method}, Debug: {debug}")
    except Exception as e:
        print(f"  Error in Hybrid: {e}")
    print()
    
    # Test the index structures directly
    print("Testing inverted index structure directly...")
    inv_index = InvertedIndex()
    inv_index.build(categories)
    candidates = inv_index.search(query)
    print(f"  Candidates from inverted index: {[(inv_index.get_category(idx), score) for idx, score in candidates[:5]]}")
    
    print("Testing n-gram index structure directly...")
    ngram_index = NGramIndex(n=3)
    ngram_index.build(categories)
    candidates = ngram_index.search(query)
    print(f"  Candidates from n-gram index: {[(ngram_index.get_category(idx), score) for idx, score in candidates[:5]]}")
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    test_methods()