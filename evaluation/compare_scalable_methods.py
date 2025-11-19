"""
Module to compare different scalable search methods for 100K+ categories
"""
import pandas as pd
from typing import List
from tqdm import tqdm
import numpy as np
from utils.parsing import parse_categories
from methods.method_scalable_inverted_ngram_hybrid import (
    method_inverted_index, method_ngram_index, method_hybrid
)
from methods.method_prefix_lexical_cross import method_adaptive_priority as method_adaptive_lexical
from methods.method_prefix_semantic_cross import method_adaptive_priority as method_adaptive_semantic
from methods.method_bm25_lexical_cross import method_adaptive_priority_bm25, method_optimized_pipeline
from models.semantic_encoder import SemanticEncoder
from models.cross_encoder import Qwen3CrossEncoder


def compare_scalable_methods(csv_path: str, device: str = "cpu", verbose: bool = False):
    """
    So sánh 6 phương pháp cho 100K+ categories:
    1. Adaptive Priority + Lexical (existing)
    2. Adaptive Priority + Semantic (existing)
    3. Adaptive Priority + BM25 Lexical (existing)
    4. Optimized Pipeline + BM25 (existing)
    5. Inverted Index (NEW - SCALABLE!)
    6. N-gram Index (NEW - SCALABLE!)
    7. Hybrid (Inverted + Semantic) (NEW - SCALABLE!)
    """
    print("=" * 80)
    print("🚀 SCALABLE METHODS FOR 100K+ CATEGORIES")
    print("=" * 80)

    # Load data
    df = pd.read_csv(csv_path)
    print(f"\n✅ Loaded {len(df)} test cases")

    # Initialize models
    print("\n" + "=" * 80)
    print("INITIALIZING MODELS")
    print("=" * 80)

    semantic_encoder = SemanticEncoder(device=device)
    cross_encoder = Qwen3CrossEncoder(device=device)

    results = []

    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)

    # Hyperparameters for existing methods
    ALPHA = 1.0   # Specificity weight
    BETA = 2.0    # Token overlap weight (quan trọng nhất!)
    GAMMA = 0.5   # Length weight

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        query = row['tên hàng hóa']
        expected = row['kết quả mong muốn']
        categories_str = row['danh mục']
        categories = parse_categories(categories_str)

        if verbose:
            print(f"\n--- Test {idx + 1}/{len(df)} ---")
            print(f"Query: {query}")
            print(f"Expected: {expected}")
            print(f"Total categories: {len(categories)}")

        # Method 1: Adaptive + Lexical
        pred1, score1, retrieval1, debug1 = method_adaptive_lexical(
            query, categories, cross_encoder, None, idx,
            use_semantic=False, alpha=ALPHA, beta=BETA, gamma=GAMMA
        )
        correct1 = (pred1.strip().lower() == expected.strip().lower())

        # Method 2: Adaptive + Semantic
        pred2, score2, retrieval2, debug2 = method_adaptive_semantic(
            query, categories, cross_encoder, semantic_encoder, idx,
            use_semantic=True, alpha=ALPHA, beta=BETA, gamma=GAMMA
        )
        correct2 = (pred2.strip().lower() == expected.strip().lower())

        # Method 3: Adaptive + BM25 Lexical
        pred3, score3, retrieval3, debug3 = method_adaptive_priority_bm25(
            query, categories, cross_encoder, None, idx,
            use_semantic=False, alpha=ALPHA, beta=BETA, gamma=GAMMA
        )
        correct3 = (pred3.strip().lower() == expected.strip().lower())

        # Method 4: Optimized Pipeline + BM25
        pred4, score4, retrieval4, debug4 = method_optimized_pipeline(
            query, categories, cross_encoder, None, idx,
            use_semantic=False, alpha=ALPHA, beta=BETA, gamma=GAMMA
        )
        correct4 = (pred4.strip().lower() == expected.strip().lower())

        # Method 5: Inverted Index (NEW)
        pred5, score5, retrieval5, debug5 = method_inverted_index(
            query, categories, cross_encoder, semantic_encoder, idx
        )
        correct5 = (pred5.strip().lower() == expected.strip().lower())

        # Method 6: N-gram Index (NEW)
        pred6, score6, retrieval6, debug6 = method_ngram_index(
            query, categories, cross_encoder, semantic_encoder, idx
        )
        correct6 = (pred6.strip().lower() == expected.strip().lower())

        # Method 7: Hybrid (Inverted + Semantic) (NEW)
        pred7, score7, retrieval7, debug7 = method_hybrid(
            query, categories, cross_encoder, semantic_encoder, idx
        )
        correct7 = (pred7.strip().lower() == expected.strip().lower())

        if verbose:
            print(f"  Adaptive+Lexical:  {pred1} [{score1:.3f}] {debug1} {'✅' if correct1 else '❌'}")
            print(f"  Adaptive+Semantic: {pred2} [{score2:.3f}] {debug2} {'✅' if correct2 else '❌'}")
            print(f"  Adaptive+BM25:     {pred3} [{score3:.3f}] {debug3} {'✅' if correct3 else '❌'}")
            print(f"  Optimized+BM25:    {pred4} [{score4:.3f}] {debug4} {'✅' if correct4 else '❌'}")
            print(f"  Inverted Index:    {pred5} [{score5:.3f}] {debug5} {'✅' if correct5 else '❌'}")
            print(f"  N-gram Index:      {pred6} [{score6:.3f}] {debug6} {'✅' if correct6 else '❌'}")
            print(f"  Hybrid:            {pred7} [{score7:.3f}] {debug7} {'✅' if correct7 else '❌'}")

        results.append({
            'query': query,
            'expected': expected,
            'total_categories': len(categories),

            # Existing methods
            'adaptive_lexical_pred': pred1,
            'adaptive_lexical_score': score1,
            'adaptive_lexical_correct': correct1,
            'adaptive_lexical_specificity': debug1.get('specificity', 0),
            'adaptive_lexical_overlap': debug1.get('token_overlap', 0),

            'adaptive_semantic_pred': pred2,
            'adaptive_semantic_score': score2,
            'adaptive_semantic_correct': correct2,
            'adaptive_semantic_specificity': debug2.get('specificity', 0),
            'adaptive_semantic_overlap': debug2.get('token_overlap', 0),

            'adaptive_bm25_pred': pred3,
            'adaptive_bm25_score': score3,
            'adaptive_bm25_correct': correct3,
            'adaptive_bm25_specificity': debug3.get('specificity', 0),
            'adaptive_bm25_overlap': debug3.get('token_overlap', 0),

            'optimized_bm25_pred': pred4,
            'optimized_bm25_score': score4,
            'optimized_bm25_correct': correct4,
            'optimized_bm25_bm25_score': debug4.get('bm25_score', 0),
            'optimized_bm25_overlap': debug4.get('token_overlap', 0),

            # NEW scalable methods
            'inverted_index_pred': pred5,
            'inverted_index_score': score5,
            'inverted_index_correct': correct5,
            'inverted_index_candidates': debug5.get('candidates', 0),
            'inverted_index_retrieval_score': debug5.get('retrieval_score', 0),

            'ngram_index_pred': pred6,
            'ngram_index_score': score6,
            'ngram_index_correct': correct6,
            'ngram_index_candidates': debug6.get('candidates', 0),
            'ngram_index_retrieval_score': debug6.get('retrieval_score', 0),

            'hybrid_pred': pred7,
            'hybrid_score': score7,
            'hybrid_correct': correct7,
            'hybrid_lexical_candidates': debug7.get('lexical_candidates', 0),
            'hybrid_semantic_candidates': debug7.get('semantic_candidates', 0),
            'hybrid_total_candidates': debug7.get('total_candidates', 0),
        })

    results_df = pd.DataFrame(results)

    # Calculate accuracies
    acc1 = (results_df['adaptive_lexical_correct'].sum() / len(results_df)) * 100
    acc2 = (results_df['adaptive_semantic_correct'].sum() / len(results_df)) * 100
    acc3 = (results_df['adaptive_bm25_correct'].sum() / len(results_df)) * 100
    acc4 = (results_df['optimized_bm25_correct'].sum() / len(results_df)) * 100
    acc5 = (results_df['inverted_index_correct'].sum() / len(results_df)) * 100
    acc6 = (results_df['ngram_index_correct'].sum() / len(results_df)) * 100
    acc7 = (results_df['hybrid_correct'].sum() / len(results_df)) * 100

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\n{'Method':<40} {'Accuracy':<12} {'Avg Candidates':<15} {'Correct/Total'}")
    print("-" * 90)
    print(f"{'1. Adaptive+Lexical':<40} {acc1:>6.2f}%     {'N/A':>12}        {results_df['adaptive_lexical_correct'].sum():>3}/{len(results_df)}")
    print(f"{'2. Adaptive+Semantic':<40} {acc2:>6.2f}%     {'N/A':>12}        {results_df['adaptive_semantic_correct'].sum():>3}/{len(results_df)}")
    print(f"{'3. Adaptive+BM25':<40} {acc3:>6.2f}%     {'N/A':>12}        {results_df['adaptive_bm25_correct'].sum():>3}/{len(results_df)}")
    print(f"{'4. Optimized+BM25':<40} {acc4:>6.2f}%     {'N/A':>12}        {results_df['optimized_bm25_correct'].sum():>3}/{len(results_df)}")
    print(f"{'5. Inverted Index':<40} {acc5:>6.2f}%     {results_df['inverted_index_candidates'].mean():>8.1f}        {results_df['inverted_index_correct'].sum():>3}/{len(results_df)}")
    print(f"{'6. N-gram Index':<40} {acc6:>6.2f}%     {results_df['ngram_index_candidates'].mean():>8.1f}        {results_df['ngram_index_correct'].sum():>3}/{len(results_df)}")
    print(f"{'7. Hybrid (Inverted+Semantic)':<40} {acc7:>6.2f}%     {results_df['hybrid_total_candidates'].mean():>8.1f}        {results_df['hybrid_correct'].sum():>3}/{len(results_df)}")
    print("-" * 90)

    # Print speedup analysis
    avg_total_cats = results_df['total_categories'].mean()
    print(f"\n📊 SCALABILITY ANALYSIS:")
    print(f"   Average total categories: {avg_total_cats:.0f}")
    print(f"   Inverted Index - Avg candidates checked: {results_df['inverted_index_candidates'].mean():.0f}")
    print(f"   N-gram Index - Avg candidates checked: {results_df['ngram_index_candidates'].mean():.0f}")
    print(f"   Hybrid - Avg candidates checked: {results_df['hybrid_total_candidates'].mean():.0f}")
    print(f"   Speedup: ~{avg_total_cats / results_df['hybrid_total_candidates'].mean():.0f}x faster! 🚀")

    # Calculate improvements
    improvement_5_over_1 = acc5 - acc1
    improvement_6_over_1 = acc6 - acc1
    improvement_7_over_1 = acc7 - acc1
    print(f"\n📊 Improvement vs Adaptive+Lexical:")
    print(f"   Inverted Index: {improvement_5_over_1:>+6.2f}%")
    print(f"   N-gram Index:   {improvement_6_over_1:>+6.2f}%")
    print(f"   Hybrid:         {improvement_7_over_1:>+6.2f}%")

    # Save results
    output_path = csv_path.replace('.csv', '_scalable_comparison.csv')
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Results saved to: {output_path}")

    return results_df


def compare_scalable_methods_only(csv_path: str, device: str = "cpu", verbose: bool = False):
    """
    So sánh chỉ 3 phương pháp SCALABLE cho 100K+ categories:
    1. Inverted Index
    2. N-gram Index
    3. Hybrid (Inverted + Semantic)
    """
    print("=" * 80)
    print("🚀 SCALABLE METHODS FOR 100K+ CATEGORIES (ONLY)")
    print("=" * 80)

    # Load data
    df = pd.read_csv(csv_path)
    print(f"\n✅ Loaded {len(df)} test cases")

    # Initialize models
    print("\n" + "=" * 80)
    print("INITIALIZING MODELS")
    print("=" * 80)

    semantic_encoder = SemanticEncoder(device=device)
    cross_encoder = Qwen3CrossEncoder(device=device)

    results = []

    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        query = row['tên hàng hóa']
        expected = row['kết quả mong muốn']
        categories_str = row['danh mục']
        categories = parse_categories(categories_str)

        if verbose:
            print(f"\n--- Test {idx + 1}/{len(df)} ---")
            print(f"Query: {query}")
            print(f"Expected: {expected}")
            print(f"Total categories: {len(categories)}")

        # Method 1: Inverted Index
        pred1, score1, method1, debug1 = method_inverted_index(
            query, categories, cross_encoder, semantic_encoder, idx
        )
        correct1 = (pred1.strip().lower() == expected.strip().lower())

        # Method 2: N-gram Index
        pred2, score2, method2, debug2 = method_ngram_index(
            query, categories, cross_encoder, semantic_encoder, idx
        )
        correct2 = (pred2.strip().lower() == expected.strip().lower())

        # Method 3: Hybrid
        pred3, score3, method3, debug3 = method_hybrid(
            query, categories, cross_encoder, semantic_encoder, idx
        )
        correct3 = (pred3.strip().lower() == expected.strip().lower())

        if verbose:
            print(f"  Inverted Index: {pred1} [{score1:.3f}] {debug1} {'✅' if correct1 else '❌'}")
            print(f"  N-gram Index:   {pred2} [{score2:.3f}] {debug2} {'✅' if correct2 else '❌'}")
            print(f"  Hybrid:         {pred3} [{score3:.3f}] {debug3} {'✅' if correct3 else '❌'}")

        results.append({
            'query': query,
            'expected': expected,
            'total_categories': len(categories),
            
            'inverted_pred': pred1,
            'inverted_score': score1,
            'inverted_correct': correct1,
            'inverted_candidates': debug1.get('candidates', 0),
            
            'ngram_pred': pred2,
            'ngram_score': score2,
            'ngram_correct': correct2,
            'ngram_candidates': debug2.get('candidates', 0),
            
            'hybrid_pred': pred3,
            'hybrid_score': score3,
            'hybrid_correct': correct3,
            'hybrid_candidates': debug3.get('total_candidates', 0),
        })

    results_df = pd.DataFrame(results)

    # Calculate accuracies
    acc1 = (results_df['inverted_correct'].sum() / len(results_df)) * 100
    acc2 = (results_df['ngram_correct'].sum() / len(results_df)) * 100
    acc3 = (results_df['hybrid_correct'].sum() / len(results_df)) * 100

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\n{'Method':<40} {'Accuracy':<12} {'Avg Candidates':<15} {'Correct/Total'}")
    print("-" * 90)
    print(f"{'1. Inverted Index':<40} {acc1:>6.2f}%     {results_df['inverted_candidates'].mean():>8.1f}        {results_df['inverted_correct'].sum():>3}/{len(results_df)}")
    print(f"{'2. N-gram Index':<40} {acc2:>6.2f}%     {results_df['ngram_candidates'].mean():>8.1f}        {results_df['ngram_correct'].sum():>3}/{len(results_df)}")
    print(f"{'3. Hybrid (Inverted + Semantic)':<40} {acc3:>6.2f}%     {results_df['hybrid_candidates'].mean():>8.1f}        {results_df['hybrid_correct'].sum():>3}/{len(results_df)}")
    print("-" * 90)

    print(f"\n📊 SPEEDUP ANALYSIS:")
    print(f"   Average total categories: {results_df['total_categories'].mean():.0f}")
    print(f"   Average candidates checked: ~{results_df['hybrid_candidates'].mean():.0f}")
    print(f"   Speedup: ~{results_df['total_categories'].mean() / results_df['hybrid_candidates'].mean():.0f}x faster! 🚀")

    # Save results
    output_path = csv_path.replace('.csv', '_scalable_comparison_only.csv')
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Results saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    csv_path = "test_cases.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = compare_scalable_methods_only(
        csv_path=csv_path,
        device=device,
        verbose=True
    )