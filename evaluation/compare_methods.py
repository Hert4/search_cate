"""
Module to compare different search methods
"""
import pandas as pd
from typing import List
from tqdm import tqdm
from utils.parsing import parse_categories
from methods.method_prefix_lexical_cross import method_adaptive_priority as method_adaptive_lexical
from methods.method_prefix_semantic_cross import method_adaptive_priority as method_adaptive_semantic
from methods.method_bm25_lexical_cross import method_adaptive_priority_bm25, method_optimized_pipeline
from models.semantic_encoder import SemanticEncoder
from models.cross_encoder import Qwen3CrossEncoder


def compare_adaptive_methods(csv_path: str, device: str = "cpu", verbose: bool = False):
    """
    So sánh:
    1. Adaptive Priority + Lexical
    2. Adaptive Priority + Semantic
    3. Adaptive Priority + BM25 Lexical
    4. Optimized Pipeline + BM25
    """
    print("=" * 80)
    print("🔥 ADAPTIVE PRIORITY SYSTEM (NO HARDCODED KEYWORDS!)")
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

    # Hyperparameters (có thể tune!)
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

        if verbose:
            print(f"  Adaptive+Lexical:  {pred1} [{score1:.3f}] {debug1} {'✅' if correct1 else '❌'}")
            print(f"  Adaptive+Semantic: {pred2} [{score2:.3f}] {debug2} {'✅' if correct2 else '❌'}")
            print(f"  Adaptive+BM25:     {pred3} [{score3:.3f}] {debug3} {'✅' if correct3 else '❌'}")
            print(f"  Optimized+BM25:    {pred4} [{score4:.3f}] {debug4} {'✅' if correct4 else '❌'}")

        results.append({
            'query': query,
            'expected': expected,

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
        })

    results_df = pd.DataFrame(results)

    # Calculate accuracies
    acc1 = (results_df['adaptive_lexical_correct'].sum() / len(results_df)) * 100
    acc2 = (results_df['adaptive_semantic_correct'].sum() / len(results_df)) * 100
    acc3 = (results_df['adaptive_bm25_correct'].sum() / len(results_df)) * 100
    acc4 = (results_df['optimized_bm25_correct'].sum() / len(results_df)) * 100

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\n{'Method':<50} {'Accuracy':<12} {'Correct/Total'}")
    print("-" * 80)
    print(f"{'Adaptive Priority + Lexical':<50} {acc1:>6.2f}%     {results_df['adaptive_lexical_correct'].sum():>3}/{len(results_df)}")
    print(f"{'Adaptive Priority + Semantic':<50} {acc2:>6.2f}%     {results_df['adaptive_semantic_correct'].sum():>3}/{len(results_df)}")
    print(f"{'Adaptive Priority + BM25 Lexical':<50} {acc3:>6.2f}%     {results_df['adaptive_bm25_correct'].sum():>3}/{len(results_df)}")
    print(f"{'Optimized Pipeline + BM25':<50} {acc4:>6.2f}%     {results_df['optimized_bm25_correct'].sum():>3}/{len(results_df)}")
    print("-" * 80)

    improvement_2_over_1 = acc2 - acc1
    improvement_3_over_1 = acc3 - acc1
    improvement_4_over_1 = acc4 - acc1
    improvement_4_over_2 = acc4 - acc2
    improvement_4_over_3 = acc4 - acc3
    print(f"\n📊 Improvement: {improvement_2_over_1:>+6.2f}% (Semantic vs Lexical)")
    print(f"📊 Improvement: {improvement_3_over_1:>+6.2f}% (BM25 vs Lexical)")
    print(f"📊 Improvement: {improvement_4_over_1:>+6.2f}% (Optimized vs Lexical)")
    print(f"📊 Improvement: {improvement_4_over_2:>+6.2f}% (Optimized vs Semantic)")
    print(f"📊 Improvement: {improvement_4_over_3:>+6.2f}% (Optimized vs BM25)")

    # Analyze priority scores
    print("\n" + "=" * 80)
    print("PRIORITY SCORE ANALYSIS")
    print("=" * 80)

    correct_cases = results_df[results_df['adaptive_semantic_correct']]
    wrong_cases = results_df[~results_df['adaptive_semantic_correct']]

    print(f"\nCorrect cases - Average token overlap: {correct_cases['adaptive_semantic_overlap'].mean():.2f}")
    print(f"Wrong cases   - Average token overlap: {wrong_cases['adaptive_semantic_overlap'].mean():.2f}")

    print(f"\nCorrect cases - Average specificity: {correct_cases['adaptive_semantic_specificity'].mean():.2f}")
    print(f"Wrong cases   - Average specificity: {wrong_cases['adaptive_semantic_specificity'].mean():.2f}")

    # Additional analysis for the optimized method
    correct_optimized = results_df[results_df['optimized_bm25_correct']]
    wrong_optimized = results_df[~results_df['optimized_bm25_correct']]

    print(f"\nOptimized BM25 - Average overlap: {correct_optimized['optimized_bm25_overlap'].mean():.2f} (correct) vs {wrong_optimized['optimized_bm25_overlap'].mean():.2f} (wrong)")
    print(f"Optimized BM25 - Average BM25 score: {correct_optimized['optimized_bm25_bm25_score'].mean():.2f} (correct) vs {wrong_optimized['optimized_bm25_bm25_score'].mean():.2f} (wrong)")

    # Save results
    output_path = csv_path.replace('.csv', '_adaptive_priority.csv')
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Results saved to: {output_path}")

    return results_df