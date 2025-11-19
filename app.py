# """
# Gradio client for the product-category matching system
# Allows users to select search methods and input product details
# """

# import gradio as gr
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from config.settings import DEVICE
# from models.semantic_encoder import SemanticEncoder
# from models.cross_encoder import Qwen3CrossEncoder
# from methods.method_prefix_lexical_cross import method_adaptive_priority as method_adaptive_lexical
# from methods.method_prefix_semantic_cross import method_adaptive_priority as method_adaptive_semantic
# from methods.method_bm25_lexical_cross import method_adaptive_priority_bm25, method_optimized_pipeline
# from utils.parsing import parse_categories


# class SearchApp:
#     def __init__(self):
#         self.semantic_encoder = None
#         self.cross_encoder = None
#         self.load_models()
    
#     def load_models(self):
#         """Load the required models"""
#         print("Loading models...")
#         try:
#             self.semantic_encoder = SemanticEncoder(device=DEVICE)
#             self.cross_encoder = Qwen3CrossEncoder(device=DEVICE)
#             print("Models loaded successfully!")
#         except Exception as e:
#             print(f"Error loading models: {e}")
    
#     def search_method(self, product_name, categories_str, method_choice):
#         """
#         Main search function that gets called by Gradio
#         """
#         if not product_name or not categories_str:
#             return "Please enter both product name and categories", "", ""
        
#         # Parse categories
#         try:
#             categories = parse_categories(f"[{categories_str}]")
#         except:
#             return "Error parsing categories. Please enter comma-separated values.", "", ""
        
#         if not categories:
#             return "No valid categories provided. Please enter comma-separated values.", "", ""
        
#         # Run selected method
#         idx = 0

#         if method_choice == "Method 1: Adaptive Priority + Lexical":
#             result, score, retrieval, debug = method_adaptive_lexical(
#                 product_name, categories, self.cross_encoder, None, idx,
#                 use_semantic=False
#             )
#             method_details = f"Method 1 Result: {result}<br>Score: {score:.3f}<br>Retrieval: {retrieval}<br>Specificity: {debug.get('specificity', 0):.2f}, Overlap: {debug.get('token_overlap', 0)}"
#             return method_details, str(result), f"{score:.3f}"

#         elif method_choice == "Method 2: Adaptive Priority + Semantic":
#             result, score, retrieval, debug = method_adaptive_semantic(
#                 product_name, categories, self.cross_encoder, self.semantic_encoder, idx,
#                 use_semantic=True
#             )
#             method_details = f"Method 2 Result: {result}<br>Score: {score:.3f}<br>Retrieval: {retrieval}<br>Specificity: {debug.get('specificity', 0):.2f}, Overlap: {debug.get('token_overlap', 0)}"
#             return method_details, str(result), f"{score:.3f}"

#         elif method_choice == "Method 3: Adaptive Priority + BM25 Lexical":
#             result, score, retrieval, debug = method_adaptive_priority_bm25(
#                 product_name, categories, self.cross_encoder, None, idx,
#                 use_semantic=False
#             )
#             method_details = f"Method 3 Result: {result}<br>Score: {score:.3f}<br>Retrieval: {retrieval}<br>Specificity: {debug.get('specificity', 0):.2f}, Overlap: {debug.get('token_overlap', 0)}"
#             return method_details, str(result), f"{score:.3f}"

#         elif method_choice == "Method 4: Optimized Pipeline + BM25":
#             result, score, retrieval, debug = method_optimized_pipeline(
#                 product_name, categories, self.cross_encoder, None, idx,
#                 use_semantic=False
#             )
#             method_details = f"Method 4 Result: {result}<br>Score: {score:.3f}<br>Retrieval: {retrieval}<br>BM25 Score: {debug.get('bm25_score', 0):.2f}, Overlap: {debug.get('token_overlap', 0)}"
#             return method_details, str(result), f"{score:.3f}"

#         elif method_choice == "All Methods: Compare Results":
#             # Method 1: Adaptive + Lexical
#             result1, score1, retrieval1, debug1 = method_adaptive_lexical(
#                 product_name, categories, self.cross_encoder, None, idx,
#                 use_semantic=False
#             )

#             # Method 2: Adaptive + Semantic
#             result2, score2, retrieval2, debug2 = method_adaptive_semantic(
#                 product_name, categories, self.cross_encoder, self.semantic_encoder, idx,
#                 use_semantic=True
#             )

#             # Method 3: Adaptive + BM25 Lexical
#             result3, score3, retrieval3, debug3 = method_adaptive_priority_bm25(
#                 product_name, categories, self.cross_encoder, None, idx,
#                 use_semantic=False
#             )

#             method1_details = f"Method 1: {result1} (Score: {score1:.3f}, Retrieval: {retrieval1}, Specificity: {debug1.get('specificity', 0):.2f}, Overlap: {debug1.get('token_overlap', 0)})"
#             method2_details = f"Method 2: {result2} (Score: {score2:.3f}, Retrieval: {retrieval2}, Specificity: {debug2.get('specificity', 0):.2f}, Overlap: {debug2.get('token_overlap', 0)})"
#             method3_details = f"Method 3: {result3} (Score: {score3:.3f}, Retrieval: {retrieval3}, Specificity: {debug3.get('specificity', 0):.2f}, Overlap: {debug3.get('token_overlap', 0)})"

#             # Compare results
#             comparison = f"<b>Comparison:</b><br>"
#             all_results = [(result1, score1), (result2, score2), (result3, score3)]
#             best_result, best_score = max(all_results, key=lambda x: x[1])

#             comparison += f"   → Best match: <b>{best_result}</b> (Score: {best_score:.3f})"

#             method_details = f"{method1_details}<br>{method2_details}<br>{method3_details}<br><br>{comparison}"
#             best_result = max(all_results, key=lambda x: x[1])[0]
#             best_score = f"{max(score1, score2, score3):.3f}"

#             return method_details, str(best_result), best_score

#         elif method_choice == "All 4 Methods: Compare Results":
#             # Method 1: Adaptive + Lexical
#             result1, score1, retrieval1, debug1 = method_adaptive_lexical(
#                 product_name, categories, self.cross_encoder, None, idx,
#                 use_semantic=False
#             )

#             # Method 2: Adaptive + Semantic
#             result2, score2, retrieval2, debug2 = method_adaptive_semantic(
#                 product_name, categories, self.cross_encoder, self.semantic_encoder, idx,
#                 use_semantic=True
#             )

#             # Method 3: Adaptive + BM25 Lexical
#             result3, score3, retrieval3, debug3 = method_adaptive_priority_bm25(
#                 product_name, categories, self.cross_encoder, None, idx,
#                 use_semantic=False
#             )

#             # Method 4: Optimized Pipeline + BM25
#             result4, score4, retrieval4, debug4 = method_optimized_pipeline(
#                 product_name, categories, self.cross_encoder, None, idx,
#                 use_semantic=False
#             )

#             method1_details = f"Method 1: {result1} (Score: {score1:.3f}, Retrieval: {retrieval1}, Specificity: {debug1.get('specificity', 0):.2f}, Overlap: {debug1.get('token_overlap', 0)})"
#             method2_details = f"Method 2: {result2} (Score: {score2:.3f}, Retrieval: {retrieval2}, Specificity: {debug2.get('specificity', 0):.2f}, Overlap: {debug2.get('token_overlap', 0)})"
#             method3_details = f"Method 3: {result3} (Score: {score3:.3f}, Retrieval: {retrieval3}, Specificity: {debug3.get('specificity', 0):.2f}, Overlap: {debug3.get('token_overlap', 0)})"
#             method4_details = f"Method 4: {result4} (Score: {score4:.3f}, Retrieval: {retrieval4}, BM25 Score: {debug4.get('bm25_score', 0):.2f}, Overlap: {debug4.get('token_overlap', 0)})"

#             # Compare results
#             comparison = f"<b>Comparison:</b><br>"
#             all_results = [(result1, score1), (result2, score2), (result3, score3), (result4, score4)]
#             best_result, best_score = max(all_results, key=lambda x: x[1])

#             comparison += f"   → Best match: <b>{best_result}</b> (Score: {best_score:.3f})"

#             method_details = f"{method1_details}<br>{method2_details}<br>{method3_details}<br>{method4_details}<br><br>{comparison}"
#             best_result = max(all_results, key=lambda x: x[1])[0]
#             best_score = f"{max(score1, score2, score3, score4):.3f}"

#             return method_details, str(best_result), best_score
#         else:  # Backwards compatibility for "Both Methods: Compare Results"
#             # Method 1: Adaptive + Lexical
#             result1, score1, retrieval1, debug1 = method_adaptive_lexical(
#                 product_name, categories, self.cross_encoder, None, idx,
#                 use_semantic=False
#             )

#             # Method 2: Adaptive + Semantic
#             result2, score2, retrieval2, debug2 = method_adaptive_semantic(
#                 product_name, categories, self.cross_encoder, self.semantic_encoder, idx,
#                 use_semantic=True
#             )

#             method1_details = f"Method 1: {result1} (Score: {score1:.3f}, Retrieval: {retrieval1}, Specificity: {debug1.get('specificity', 0):.2f}, Overlap: {debug1.get('token_overlap', 0)})"
#             method2_details = f"Method 2: {result2} (Score: {score2:.3f}, Retrieval: {retrieval2}, Specificity: {debug2.get('specificity', 0):.2f}, Overlap: {debug2.get('token_overlap', 0)})"

#             # Compare results
#             comparison = f"<b>Comparison:</b><br>"
#             all_results = [(result1, score1), (result2, score2)]
#             best_result, best_score = max(all_results, key=lambda x: x[1])

#             comparison += f"   → Best match: <b>{best_result}</b> (Score: {best_score:.3f})"

#             method_details = f"{method1_details}<br>{method2_details}<br><br>{comparison}"
#             best_result = max([(result1, score1), (result2, score2)], key=lambda x: x[1])[0]
#             best_score = f"{max(score1, score2):.3f}"

#             return method_details, str(best_result), best_score


# def create_interface():
#     """Create the Gradio interface"""
#     app = SearchApp()
    
#     with gr.Blocks(title="Product-Category Matching System") as interface:
#         gr.Markdown("""
#         # 🛒 Product-Category Matching System
        
#         Select a search method and enter a product name with its possible categories to find the best match.
#         """)
        
#         with gr.Row():
#             with gr.Column():
#                 method_choice = gr.Radio(
#                     choices=[
#                         "Method 1: Adaptive Priority + Lexical",
#                         "Method 2: Adaptive Priority + Semantic",
#                         "Method 3: Adaptive Priority + BM25 Lexical",
#                         "Method 4: Optimized Pipeline + BM25",
#                         "Both Methods: Compare Results",
#                         "All Methods: Compare Results",
#                         "All 4 Methods: Compare Results"
#                     ],
#                     value="All 4 Methods: Compare Results",
#                     label="Search Method"
#                 )
                
#                 product_name = gr.Textbox(
#                     label="Product Name",
#                     placeholder="Enter product name (e.g., iPhone 13 Pro Max)"
#                 )
                
#                 categories_str = gr.Textbox(
#                     label="Categories",
#                     placeholder="Enter categories separated by commas (e.g., 'Điện tử, Điện lạnh, Thiết bị số, Máy tính, Laptop, Điện thoại, Smartphone')",
#                     lines=3
#                 )
                
#                 submit_btn = gr.Button("Search", variant="primary")
            
#             with gr.Column():
#                 method_details = gr.HTML(label="Method Details")
#                 best_result = gr.Textbox(label="Best Result", interactive=False)
#                 best_score = gr.Textbox(label="Confidence Score", interactive=False)
        
#         submit_btn.click(
#             fn=app.search_method,
#             inputs=[product_name, categories_str, method_choice],
#             outputs=[method_details, best_result, best_score]
#         )
        
#         gr.Examples(
#             examples = [
#                 ["Điện thoại Iphone 13", "Điện tử, Điện lạnh, Thiết bị số, Máy tính, Laptop, Điện thoại, Smartphone", "All 4 Methods: Compare Results"],
#                 ["MacBook Pro M2", "Điện tử, Điện lạnh, Thiết bị số, Máy tính, Laptop, Điện thoại, Smartphone", "All 4 Methods: Compare Results"],
#             ],

#             inputs=[product_name, categories_str, method_choice],
#             outputs=[method_details, best_result, best_score],
#             label="Example Inputs"
#         )
    
#     return interface


# if __name__ == "__main__":
#     interface = create_interface()
#     interface.launch(share=True, debug = True)


import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import os

# Cấu hình môi trường
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config.settings import DEVICE
from models.semantic_encoder import SemanticEncoder
from models.cross_encoder import Qwen3CrossEncoder
from methods.method_prefix_lexical_cross import method_adaptive_priority as method_adaptive_lexical
from methods.method_prefix_semantic_cross import method_adaptive_priority as method_adaptive_semantic
from methods.method_bm25_lexical_cross import method_adaptive_priority_bm25, method_optimized_pipeline
from utils.parsing import parse_categories

class SearchApp:
    def __init__(self):
        self.semantic_encoder = None
        self.cross_encoder = None
        self.load_models()
    
    def load_models(self):
        """Load the required models"""
        print("Loading models...")
        try:
            self.semantic_encoder = SemanticEncoder(device=DEVICE)
            self.cross_encoder = Qwen3CrossEncoder(device=DEVICE)
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")

    # --- LOGIC SEARCH ĐƠN LẺ (CŨ) ---
    def search_method(self, product_name, categories_str, method_choice):
        if not product_name or not categories_str:
            return "Please enter both product name and categories", "", ""
        
        try:
            categories = parse_categories(f"[{categories_str}]")
        except:
            return "Error parsing categories.", "", ""
        
        if not categories:
            return "No valid categories provided.", "", ""
        
        idx = 0
        
        # Logic chạy từng method (giữ nguyên logic cũ của bạn để ngắn gọn)
        # Tôi tóm tắt lại phần gọi hàm để code gọn hơn
        
        res_list = []
        
        # Method 1
        if "Method 1" in method_choice or "All" in method_choice:
            r, s, ret, d = method_adaptive_lexical(product_name, categories, self.cross_encoder, None, idx, use_semantic=False)
            res_list.append(("Method 1 (Lexical)", r, s, ret, d))

        # Method 2
        if "Method 2" in method_choice or "All" in method_choice:
            r, s, ret, d = method_adaptive_semantic(product_name, categories, self.cross_encoder, self.semantic_encoder, idx, use_semantic=True)
            res_list.append(("Method 2 (Semantic)", r, s, ret, d))
            
        # Method 3
        if "Method 3" in method_choice or "All" in method_choice:
            r, s, ret, d = method_adaptive_priority_bm25(product_name, categories, self.cross_encoder, None, idx, use_semantic=False)
            res_list.append(("Method 3 (BM25)", r, s, ret, d))

        # Method 4
        if "Method 4" in method_choice or "All 4" in method_choice:
            r, s, ret, d = method_optimized_pipeline(product_name, categories, self.cross_encoder, None, idx, use_semantic=False)
            res_list.append(("Method 4 (Optimized)", r, s, ret, d))
        
        # Format Output
        details_html = ""
        best_res = ("", -1)
        
        for name, r, s, ret, d in res_list:
            # Lấy thông tin debug tùy method
            extra_info = ""
            if "bm25_score" in d:
                extra_info = f"BM25: {d.get('bm25_score', 0):.2f}"
            else:
                extra_info = f"Specificity: {d.get('specificity', 0):.2f}"
            
            details_html += f"<b>{name}</b>: {r} <br>(Score: {s:.3f}, Retrieval: {ret}, Overlap: {d.get('token_overlap', 0)}, {extra_info})<br><br>"
            
            if s > best_res[1]:
                best_res = (r, s)

        if len(res_list) > 1:
             details_html += f"<b>Comparison:</b><br>   → Best match: <b>{best_res[0]}</b> (Score: {best_res[1]:.3f})"

        return details_html, str(best_res[0]), f"{best_res[1]:.3f}"

    # --- LOGIC BENCHMARK MỚI ---
    def run_benchmark_ui(self, file_obj, progress=gr.Progress()):
        """
        Hàm xử lý file CSV và chạy benchmark, trả về biểu đồ và bảng
        """
        if file_obj is None:
            return None, "Please upload a CSV file.", pd.DataFrame()

        # 1. Load Data
        try:
            df = pd.read_csv(file_obj.name)
            required_cols = ['tên hàng hóa', 'kết quả mong muốn', 'danh mục']
            if not all(col in df.columns for col in required_cols):
                return None, f"CSV must contain columns: {required_cols}", pd.DataFrame()
        except Exception as e:
            return None, f"Error reading CSV: {e}", pd.DataFrame()

        results = []
        
        # Hyperparameters
        ALPHA, BETA, GAMMA = 1.0, 2.0, 0.5

        # 2. Run Loop with Progress Bar
        for idx, row in progress.tqdm(df.iterrows(), total=len(df), desc="Running Benchmarks"):
            query = row['tên hàng hóa']
            expected = row['kết quả mong muốn']
            categories_str = row['danh mục']
            categories = parse_categories(categories_str)

            # Run 4 Methods
            # M1
            p1, s1, _, _ = method_adaptive_lexical(query, categories, self.cross_encoder, None, idx, use_semantic=False, alpha=ALPHA, beta=BETA, gamma=GAMMA)
            c1 = (p1.strip().lower() == expected.strip().lower())
            
            # M2
            p2, s2, _, _ = method_adaptive_semantic(query, categories, self.cross_encoder, self.semantic_encoder, idx, use_semantic=True, alpha=ALPHA, beta=BETA, gamma=GAMMA)
            c2 = (p2.strip().lower() == expected.strip().lower())

            # M3
            p3, s3, _, _ = method_adaptive_priority_bm25(query, categories, self.cross_encoder, None, idx, use_semantic=False, alpha=ALPHA, beta=BETA, gamma=GAMMA)
            c3 = (p3.strip().lower() == expected.strip().lower())

            # M4
            p4, s4, _, _ = method_optimized_pipeline(query, categories, self.cross_encoder, None, idx, use_semantic=False, alpha=ALPHA, beta=BETA, gamma=GAMMA)
            c4 = (p4.strip().lower() == expected.strip().lower())

            results.append({
                'Query': query,
                'Expected': expected,
                'M1_Pred': p1, 'M1_Correct': c1,
                'M2_Pred': p2, 'M2_Correct': c2,
                'M3_Pred': p3, 'M3_Correct': c3,
                'M4_Pred': p4, 'M4_Correct': c4,
            })

        res_df = pd.DataFrame(results)

        # 3. Calculate Statistics
        acc1 = res_df['M1_Correct'].mean() * 100
        acc2 = res_df['M2_Correct'].mean() * 100
        acc3 = res_df['M3_Correct'].mean() * 100
        acc4 = res_df['M4_Correct'].mean() * 100

        # 4. Create Summary Report (Markdown)
        summary_md = f"""
        ### 📊 Benchmark Results Summary
        *Total Test Cases: {len(res_df)}*
        
        | Method | Accuracy | Correct/Total | Improvement (vs M1) |
        | :--- | :--- | :--- | :--- |
        | **1. Adaptive Lexical** | **{acc1:.2f}%** | {res_df['M1_Correct'].sum()}/{len(res_df)} | - |
        | **2. Adaptive Semantic** | **{acc2:.2f}%** | {res_df['M2_Correct'].sum()}/{len(res_df)} | {acc2-acc1:+.2f}% |
        | **3. Adaptive BM25** | **{acc3:.2f}%** | {res_df['M3_Correct'].sum()}/{len(res_df)} | {acc3-acc1:+.2f}% |
        | **4. Optimized BM25** | **{acc4:.2f}%** | {res_df['M4_Correct'].sum()}/{len(res_df)} | {acc4-acc1:+.2f}% |
        """

        # 5. Create Visualization (Matplotlib Chart)
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = ['M1: Lexical', 'M2: Semantic', 'M3: BM25', 'M4: Optimized']
        accuracies = [acc1, acc2, acc3, acc4]
        colors = ['#dda0dd', '#87cefa', '#90ee90', '#ff7f50'] # Colors for visualization

        bars = ax.bar(methods, accuracies, color=colors)
        
        # Add text on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Comparison of Method Accuracies (N={len(res_df)})')
        ax.set_ylim(0, 110)
        plt.tight_layout()

        return fig, summary_md, res_df


def create_interface():
    app = SearchApp()
    
    with gr.Blocks(title="Product-Category Matching System") as interface:
        gr.Markdown("# 🛒 Intelligent Product Categorization System")
        
        # --- SỬ DỤNG TABS ĐỂ CHIA GIAO DIỆN ---
        with gr.Tabs():
            
            # TAB 1: TÌM KIẾM ĐƠN (Giao diện cũ)
            with gr.TabItem("🔍 Single Search"):
                with gr.Row():
                    with gr.Column(scale=1):
                        method_choice = gr.Radio(
                            choices=[
                                "Method 1: Adaptive Priority + Lexical",
                                "Method 2: Adaptive Priority + Semantic",
                                "Method 3: Adaptive Priority + BM25 Lexical",
                                "Method 4: Optimized Pipeline + BM25",
                                "All 4 Methods: Compare Results"
                            ],
                            value="All 4 Methods: Compare Results",
                            label="Search Method"
                        )
                        product_name = gr.Textbox(label="Product Name", placeholder="Enter product name (e.g., iPhone 13)")
                        categories_str = gr.Textbox(label="Categories", placeholder="Cat1, Cat2, Cat3...", lines=3)
                        submit_btn = gr.Button("Search", variant="primary")
                    
                    with gr.Column(scale=1):
                        method_details = gr.HTML(label="Method Details")
                        with gr.Row():
                            best_result = gr.Textbox(label="Best Match", interactive=False)
                            best_score = gr.Textbox(label="Confidence Score", interactive=False)

                # Event Handler Tab 1
                submit_btn.click(
                    fn=app.search_method,
                    inputs=[product_name, categories_str, method_choice],
                    outputs=[method_details, best_result, best_score]
                )
                
                gr.Examples(
                    examples = [
                        ["Điện thoại Iphone 13", "Điện tử, Điện lạnh, Thiết bị số, Máy tính, Laptop, Điện thoại, Smartphone", "All 4 Methods: Compare Results"],
                    ],
                    inputs=[product_name, categories_str, method_choice],
                    label="Example Inputs"
                )

            # TAB 2: BENCHMARK (Giao diện mới)
            with gr.TabItem("📊 Benchmark Evaluation"):
                gr.Markdown("### Upload a CSV file to compare all methods on a testset.")
                gr.Markdown("The CSV must have columns: `tên hàng hóa`, `kết quả mong muốn`, `danh mục`")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(label="Upload Test CSV", file_types=[".csv"])
                        run_bench_btn = gr.Button("🚀 Run Benchmark", variant="primary")
                    
                    with gr.Column(scale=2):
                        # Output 1: Biểu đồ
                        plot_output = gr.Plot(label="Accuracy Visualization")
                
                # Output 2: Bảng thống kê text
                summary_output = gr.Markdown(label="Summary Stats")
                
                # Output 3: Chi tiết dữ liệu (Cho phép scroll và xem lỗi sai)
                details_output = gr.DataFrame(
                    label="Detailed Results", 
                    headers=["Query", "Expected", "M1_Pred", "M1_Correct", "M4_Pred", "M4_Correct"],
                    interactive=False
                )

                # Event Handler Tab 2
                run_bench_btn.click(
                    fn=app.run_benchmark_ui,
                    inputs=[file_input],
                    outputs=[plot_output, summary_output, details_output]
                )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True, debug=True)