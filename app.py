import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import os


from config.settings import DEVICE
from models.semantic_encoder import SemanticEncoder
from models.cross_encoder import Qwen3CrossEncoder
from methods.method_prefix_lexical_cross import (
    method_adaptive_priority as method_adaptive_lexical,
)
from methods.method_prefix_semantic_cross import (
    method_adaptive_priority as method_adaptive_semantic,
)
from methods.method_bm25_lexical_cross import (
    method_adaptive_priority_bm25,
    method_optimized_pipeline,
)
from methods.method_scalable_inverted_ngram_hybrid import (
    method_inverted_index,
    method_ngram_index,
    method_hybrid,
)
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
            r, s, ret, d = method_adaptive_lexical(
                product_name,
                categories,
                self.cross_encoder,
                None,
                idx,
                use_semantic=False,
            )
            res_list.append(("Method 1 (Lexical)", r, s, ret, d))

        # Method 2
        if "Method 2" in method_choice or "All" in method_choice:
            r, s, ret, d = method_adaptive_semantic(
                product_name,
                categories,
                self.cross_encoder,
                self.semantic_encoder,
                idx,
                use_semantic=True,
            )
            res_list.append(("Method 2 (Semantic)", r, s, ret, d))

        # Method 3
        if "Method 3" in method_choice or "All" in method_choice:
            r, s, ret, d = method_adaptive_priority_bm25(
                product_name,
                categories,
                self.cross_encoder,
                None,
                idx,
                use_semantic=False,
            )
            res_list.append(("Method 3 (BM25)", r, s, ret, d))

        # Method 4
        if "Method 4" in method_choice or "All 4" in method_choice:
            r, s, ret, d = method_optimized_pipeline(
                product_name,
                categories,
                self.cross_encoder,
                None,
                idx,
                use_semantic=False,
            )
            res_list.append(("Method 4 (Optimized)", r, s, ret, d))

        # Method 5: Inverted Index
        if "Method 5" in method_choice or "All 7" in method_choice:
            r, s, ret, d = method_inverted_index(
                product_name, categories, self.cross_encoder, self.semantic_encoder, idx
            )
            res_list.append(("Method 5 (Inverted Index)", r, s, ret, d))

        # Method 6: N-gram Index
        if "Method 6" in method_choice or "All 7" in method_choice:
            r, s, ret, d = method_ngram_index(
                product_name, categories, self.cross_encoder, self.semantic_encoder, idx
            )
            res_list.append(("Method 6 (N-gram Index)", r, s, ret, d))

        # Method 7: Hybrid
        if "Method 7" in method_choice or "All 7" in method_choice:
            r, s, ret, d = method_hybrid(
                product_name, categories, self.cross_encoder, self.semantic_encoder, idx
            )
            res_list.append(("Method 7 (Hybrid)", r, s, ret, d))

        # Format Output
        details_html = ""
        best_res = ("", -1)

        for name, r, s, ret, d in res_list:
            # Lấy thông tin debug tùy method
            extra_info = ""
            if "Method 5" in name or "Method 6" in name or "Method 7" in name:
                # For scalable methods
                if "retrieval_score" in d:
                    extra_info = f"Retrieval Score: {d.get('retrieval_score', 0):.2f}, Candidates: {d.get('candidates', 0)}"
                elif "total_candidates" in d:
                    extra_info = f"Total Candidates: {d.get('total_candidates', 0)}"
                else:
                    extra_info = f"Candidates: {d.get('candidates', 0)}"
            elif "bm25_score" in d:
                extra_info = f"BM25: {d.get('bm25_score', 0):.2f}"
            elif "specificity" in d:
                extra_info = f"Specificity: {d.get('specificity', 0):.2f}"
            else:
                extra_info = f"Token Overlap: {d.get('token_overlap', 0)}"

            details_html += f"<b>{name}</b>: {r} <br>(Score: {s:.3f}, Retrieval: {ret}, Overlap: {d.get('token_overlap', 0) if 'token_overlap' in d else 0}, {extra_info})<br><br>"

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
            required_cols = ["tên hàng hóa", "kết quả mong muốn", "danh mục"]
            if not all(col in df.columns for col in required_cols):
                return (
                    None,
                    f"CSV must contain columns: {required_cols}",
                    pd.DataFrame(),
                )
        except Exception as e:
            return None, f"Error reading CSV: {e}", pd.DataFrame()

        results = []

        # Hyperparameters
        ALPHA, BETA, GAMMA = 1.0, 2.0, 0.5

        # 2. Run Loop with Progress Bar
        for idx, row in progress.tqdm(
            df.iterrows(), total=len(df), desc="Running Benchmarks"
        ):
            query = row["tên hàng hóa"]
            expected = row["kết quả mong muốn"]
            categories_str = row["danh mục"]
            categories = parse_categories(categories_str)

            # Run 7 Methods
            # M1
            p1, s1, _, _ = method_adaptive_lexical(
                query,
                categories,
                self.cross_encoder,
                None,
                idx,
                use_semantic=False,
                alpha=ALPHA,
                beta=BETA,
                gamma=GAMMA,
            )
            c1 = p1.strip().lower() == expected.strip().lower()

            # M2
            p2, s2, _, _ = method_adaptive_semantic(
                query,
                categories,
                self.cross_encoder,
                self.semantic_encoder,
                idx,
                use_semantic=True,
                alpha=ALPHA,
                beta=BETA,
                gamma=GAMMA,
            )
            c2 = p2.strip().lower() == expected.strip().lower()

            # M3
            p3, s3, _, _ = method_adaptive_priority_bm25(
                query,
                categories,
                self.cross_encoder,
                None,
                idx,
                use_semantic=False,
                alpha=ALPHA,
                beta=BETA,
                gamma=GAMMA,
            )
            c3 = p3.strip().lower() == expected.strip().lower()

            # M4
            p4, s4, _, _ = method_optimized_pipeline(
                query,
                categories,
                self.cross_encoder,
                None,
                idx,
                use_semantic=False,
                alpha=ALPHA,
                beta=BETA,
                gamma=GAMMA,
            )
            c4 = p4.strip().lower() == expected.strip().lower()

            # M5
            p5, s5, _, _ = method_inverted_index(
                query, categories, self.cross_encoder, self.semantic_encoder, idx
            )
            c5 = p5.strip().lower() == expected.strip().lower()

            # M6
            p6, s6, _, _ = method_ngram_index(
                query, categories, self.cross_encoder, self.semantic_encoder, idx
            )
            c6 = p6.strip().lower() == expected.strip().lower()

            # M7
            p7, s7, _, _ = method_hybrid(
                query, categories, self.cross_encoder, self.semantic_encoder, idx
            )
            c7 = p7.strip().lower() == expected.strip().lower()

            results.append(
                {
                    "Query": query,
                    "Expected": expected,
                    "M1_Pred": p1,
                    "M1_Correct": c1,
                    "M2_Pred": p2,
                    "M2_Correct": c2,
                    "M3_Pred": p3,
                    "M3_Correct": c3,
                    "M4_Pred": p4,
                    "M4_Correct": c4,
                    "M5_Pred": p5,
                    "M5_Correct": c5,
                    "M6_Pred": p6,
                    "M6_Correct": c6,
                    "M7_Pred": p7,
                    "M7_Correct": c7,
                }
            )

        res_df = pd.DataFrame(results)

        # 3. Calculate Statistics
        acc1 = res_df["M1_Correct"].mean() * 100
        acc2 = res_df["M2_Correct"].mean() * 100
        acc3 = res_df["M3_Correct"].mean() * 100
        acc4 = res_df["M4_Correct"].mean() * 100
        acc5 = res_df["M5_Correct"].mean() * 100
        acc6 = res_df["M6_Correct"].mean() * 100
        acc7 = res_df["M7_Correct"].mean() * 100

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
        | **5. Inverted Index** | **{acc5:.2f}%** | {res_df['M5_Correct'].sum()}/{len(res_df)} | {acc5-acc1:+.2f}% |
        | **6. N-gram Index** | **{acc6:.2f}%** | {res_df['M6_Correct'].sum()}/{len(res_df)} | {acc6-acc1:+.2f}% |
        | **7. Hybrid** | **{acc7:.2f}%** | {res_df['M7_Correct'].sum()}/{len(res_df)} | {acc7-acc1:+.2f}% |
        """

        # 5. Create Visualization (Matplotlib Chart)
        fig, ax = plt.subplots(figsize=(12, 6))
        methods = [
            "M1: Lexical",
            "M2: Semantic",
            "M3: BM25",
            "M4: Optimized",
            "M5: Inverted",
            "M6: N-gram",
            "M7: Hybrid",
        ]
        accuracies = [acc1, acc2, acc3, acc4, acc5, acc6, acc7]
        colors = [
            "#dda0dd",
            "#87cefa",
            "#90ee90",
            "#ff7f50",
            "#f0e68c",
            "#dda0dd",
            "#add8e6",
        ]  # Colors for visualization

        bars = ax.bar(methods, accuracies, color=colors)

        # Add text on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 1,
                f"{yval:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Comparison of Method Accuracies (N={len(res_df)})")
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
                                "Method 5: Inverted Index",
                                "Method 6: N-gram Index",
                                "Method 7: Hybrid (Inverted + Semantic)",
                                "All 4 Methods: Compare Results",
                                "All 7 Methods: Compare Results",
                            ],
                            value="All 7 Methods: Compare Results",
                            label="Search Method",
                        )
                        product_name = gr.Textbox(
                            label="Product Name",
                            placeholder="Enter product name (e.g., iPhone 13)",
                        )
                        categories_str = gr.Textbox(
                            label="Categories",
                            placeholder="Cat1, Cat2, Cat3...",
                            lines=3,
                        )
                        submit_btn = gr.Button("Search", variant="primary")

                    with gr.Column(scale=1):
                        method_details = gr.HTML(label="Method Details")
                        with gr.Row():
                            best_result = gr.Textbox(
                                label="Best Match", interactive=False
                            )
                            best_score = gr.Textbox(
                                label="Confidence Score", interactive=False
                            )

                # Event Handler Tab 1
                submit_btn.click(
                    fn=app.search_method,
                    inputs=[product_name, categories_str, method_choice],
                    outputs=[method_details, best_result, best_score],
                )

                gr.Examples(
                    examples=[
                        [
                            "Điện thoại Iphone 13",
                            "Điện tử, Điện lạnh, Thiết bị số, Máy tính, Laptop, Điện thoại, Smartphone",
                            "All 4 Methods: Compare Results",
                        ],
                    ],
                    inputs=[product_name, categories_str, method_choice],
                    label="Example Inputs",
                )

            # TAB 2: BENCHMARK (Giao diện mới)
            with gr.TabItem("📊 Benchmark Evaluation"):
                gr.Markdown(
                    "### Upload a CSV file to compare all methods on a testset."
                )
                gr.Markdown(
                    "The CSV must have columns: `tên hàng hóa`, `kết quả mong muốn`, `danh mục`"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="Upload Test CSV", file_types=[".csv"]
                        )
                        run_bench_btn = gr.Button("🚀 Run Benchmark", variant="primary")

                    with gr.Column(scale=2):
                        # Output 1: Biểu đồ
                        plot_output = gr.Plot(label="Accuracy Visualization")

                # Output 2: Bảng thống kê text
                summary_output = gr.Markdown(label="Summary Stats")

                # Output 3: Chi tiết dữ liệu (Cho phép scroll và xem lỗi sai)
                details_output = gr.DataFrame(
                    label="Detailed Results",
                    headers=[
                        "Query",
                        "Expected",
                        "M1_Pred",
                        "M1_Correct",
                        "M4_Pred",
                        "M4_Correct",
                    ],
                    interactive=False,
                )

                # Event Handler Tab 2
                run_bench_btn.click(
                    fn=app.run_benchmark_ui,
                    inputs=[file_input],
                    outputs=[plot_output, summary_output, details_output],
                )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True, debug=True)
