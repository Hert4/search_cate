# Intelligent Product Categorization System

This project implements multiple approaches for intelligent product categorization, matching product names to the most appropriate categories from a large set of possible categories. The system is designed to handle both small-scale and large-scale (100K+ categories) scenarios with different algorithmic approaches.

## 📋 Table of Contents

- [Overview](#overview)
- [Methods Overview](#methods-overview)
- [Detailed Method Descriptions](#detailed-method-descriptions)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [Performance Comparison](#performance-comparison)
- [Installation](#installation)

## Overview

The system provides 7 different methods for product categorization, each optimized for different use cases and data characteristics. The core idea is to match a product name to the most relevant category from a given list of categories using various search and ranking techniques.

## Methods Overview

The project implements 7 distinct approaches:

1. **Adaptive Priority + Lexical** - Uses prefix tree with lexical search
2. **Adaptive Priority + Semantic** - Uses prefix tree with semantic search  
3. **Adaptive Priority + BM25 Lexical** - Uses BM25 algorithm for lexical matching
4. **Optimized Pipeline + BM25** - Enhanced BM25 with cross-encoder reranking
5. **Inverted Index** - Scalable approach using inverted index
6. **N-gram Index** - Scalable approach using n-gram matching
7. **Hybrid** - Combines inverted index and semantic search

## Detailed Method Descriptions

### 1. Adaptive Priority + Lexical (`method_prefix_lexical_cross.py`)

**Approach**: Combines prefix tree matching with lexical search, enhanced with adaptive priority calculation

**Pipeline**:
1. Calculate adaptive priority for all categories based on specificity, token overlap, and length
2. Build prefix tree from prioritized categories
3. Use prefix matching to find initial candidates
4. If no prefix match found, use lexical search
5. Apply cross-encoder reranking to determine best match

**Use Case**: Good for scenarios where categories have common prefixes with the query

**Minh họa cách hoạt động**:
```
Query: "iPhone 13 Pro Max"
Categories: ["iPhone", "iPhone 13", "iPhone 13 Pro", "iPhone 13 Pro Max", "Điện thoại", "Điện tử"]

Bước 1 - Tính Priority:
- "iPhone": specificity=0.1, overlap=1, length=1 → priority=2.6
- "iPhone 13": specificity=0.2, overlap=2, length=2 → priority=5.4
- "iPhone 13 Pro": specificity=0.3, overlap=3, length=3 → priority=8.3
- "iPhone 13 Pro Max": specificity=0.4, overlap=4, length=4 → priority=11.4
- "Điện thoại": specificity=0.5, overlap=0, length=2 → priority=1.5
- "Điện tử": specificity=0.6, overlap=0, length=2 → priority=1.6

Bước 2 - Build Prefix Tree:
                      root
                       |
                     iPhone ──┬── "iPhone" (priority=2.6)
                       |     └── 13 ──┬── "iPhone 13" (priority=5.4)
                       |              └── Pro ──┬── "iPhone 13 Pro" (priority=8.3)
                       |                        └── Max ── "iPhone 13 Pro Max" (priority=11.4)

Bước 3 - Prefix Matching:
- Query "iPhone 13 Pro Max" → đi theo đường dẫn tree → tìm thấy "iPhone 13 Pro Max"
- Lấy top 50 candidates theo priority (trong trường hợp này chỉ lấy các node phù hợp)

Bước 4 - Cross-Encoder Reranking:
- So sánh: "iPhone 13 Pro Max" vs "iPhone 13 Pro Max"
- Cross-encoder xác nhận độ tương đồng cao nhất → Kết quả: "iPhone 13 Pro Max"
```

### 2. Adaptive Priority + Semantic (`method_prefix_semantic_cross.py`)

**Approach**: Combines prefix tree matching with semantic search using dense embeddings

**Pipeline**:
1. Calculate adaptive priority for all categories
2. Build prefix tree from prioritized categories
3. Use prefix matching to find initial candidates
4. If no prefix match found, use semantic search
5. Apply cross-encoder reranking

**Use Case**: Effective when semantic meaning is important and categories may not share exact tokens with query

**Minh họa cách hoạt động**:
```
Query: "Điện thoại thông minh Samsung Galaxy"
Categories: ["Mobile Devices", "Điện thoại Android", "Điện tử tiêu dùng", "Thiết bị di động"]

Bước 1 - Tính Priority:
- "Mobile Devices": specificity=0.8, overlap=1, length=2 → priority=2.8
- "Điện thoại Android": specificity=0.9, overlap=1, length=2 → priority=2.9
- "Điện tử tiêu dùng": specificity=0.5, overlap=0, length=2 → priority=1.5
- "Thiết bị di động": specificity=0.6, overlap=0, length=2 → priority=1.6

Bước 2 - Prefix Matching:
- Query không match với prefix tree (không có node nào bắt đầu với "Điện thoại thông minh...")

Bước 3 - Semantic Search (khi không có prefix match):
- Encode query: "Điện thoại thông minh Samsung Galaxy" → vector A
- Encode categories: ["Mobile Devices", "Điện thoại Android", ...] → vector [B, C, ...]
- Tính similarity: cos(A, B), cos(A, C), ...
- Gần nhất: "Điện thoại Android" (vector C) - tương tự về ngữ nghĩa

Bước 4 - Cross-Encoder Reranking:
- Pair: ["Điện thoại thông minh Samsung Galaxy", "Điện thoại Android"]
- Cross-encoder xác nhận mối liên kết ngữ nghĩa → Kết quả: "Điện thoại Android"
```

### 3. Adaptive Priority + BM25 Lexical (`method_bm25_lexical_cross.py`)

**Approach**: Uses BM25 algorithm for more sophisticated lexical matching

**Pipeline**:
1. Calculate adaptive priority for all categories
2. Use BM25 algorithm for initial retrieval (top 50 candidates)
3. Apply cross-encoder reranking

**Use Case**: Better for scenarios with varied query lengths and token distributions

**Minh họa cách hoạt động**:
```
Query: "Sữa rửa mặt tạo bọt La Roche-Posay cho da nhạy cảm"
Categories: ["Sữa rửa mặt", "Chăm sóc da", "Mỹ phẩm", "Sữa rửa mặt cho da nhạy cảm", "La Roche-Posay"]

Bước 1 - Tính Priority:
- "Sữa rửa mặt": specificity=0.3, overlap=2, length=3 → priority=5.3
- "Chăm sóc da": specificity=0.4, overlap=0, length=2 → priority=1.4
- "Mỹ phẩm": specificity=0.2, overlap=0, length=1 → priority=0.2
- "Sữa rửa mặt cho da nhạy cảm": specificity=0.7, overlap=5, length=5 → priority=12.9
- "La Roche-Posay": specificity=0.9, overlap=1, length=2 → priority=3.9

Bước 2 - BM25 Retrieval:
- Tính BM25 score cho từng category:
  - "Sữa rửa mặt": TF-IDF based score = 0.4
  - "Chăm sóc da": TF-IDF based score = 0.1
  - "Mỹ phẩm": TF-IDF based score = 0.05
  - "Sữa rửa mặt cho da nhạy cảm": TF-IDF based score = 1.2
  - "La Roche-Posay": TF-IDF based score = 0.8

- Sắp xếp: ["Sữa rửa mặt cho da nhạy cảm", "La Roche-Posay", "Sữa rửa mặt", ...]
- Lấy top 50 candidates (trong trường hợp này là tất cả)

Bước 3 - Cross-Encoder Reranking:
- So sánh từng cặp với cross-encoder
- ["Sữa rửa mặt tạo bọt La Roche-Posay cho da nhạy cảm", "Sữa rửa mặt cho da nhạy cảm"] → score cao nhất
- Kết quả: "Sữa rửa mặt cho da nhạy cảm" (token overlap: 5, độ phù hợp cao)
```

### 4. Optimized Pipeline + BM25 (`method_bm25_lexical_cross.py`)

**Approach**: Enhanced version of BM25 method with optimized retrieval

**Pipeline**:
1. Create prioritized products with auto priority calculation
2. Initialize BM25Retriever with prioritized products
3. Retrieve top 50 candidates using optimized BM25
4. Cross-encoder reranking for final ranking

**Use Case**: When you need the most optimized lexical approach

**Minh họa cách hoạt động**:
```
Query: "MacBook Pro 16 inch 2022 M1 Pro"
Categories: ["Laptop", "MacBook", "MacBook Pro", "MacBook Pro 16 inch", "MacBook Pro 16 inch 2022", "MacBook Pro 16 inch 2022 M1 Pro", "Máy tính"]

Bước 1 - Tạo prioritized products:
- Tính priority cho từng category dựa trên query
- ["MacBook Pro 16 inch 2022 M1 Pro": priority=15.2, "MacBook Pro 16 inch 2022": priority=13.1, ...]

Bước 2 - Initialize BM25Retriever:
- Index các categories đã được sắp xếp theo priority
- BM25Retriever([prioritized_categories])

Bước 3 - BM25 Search:
- Query: "MacBook Pro 16 inch 2022 M1 Pro"
- BM25 tìm top 50 candidates, xếp theo relevance score
- Kết quả: ["MacBook Pro 16 inch 2022 M1 Pro", "MacBook Pro 16 inch 2022", "MacBook Pro 16 inch", ...]

Bước 4 - Cross-Encoder Reranking:
- Tính cross-encoder score cho từng cặp query-category
- ["MacBook Pro 16 inch 2022 M1 Pro", "MacBook Pro 16 inch 2022 M1 Pro"] → score cao nhất
- Kết quả: "MacBook Pro 16 inch 2022 M1 Pro"
```

### 5. Inverted Index (`method_scalable_inverted_ngram_hybrid.py`)

**Approach**: Scalable method using inverted index for efficient retrieval

**Pipeline**:
1. Build inverted index mapping tokens to category indices
2. For query, find intersection of token sets to get candidates
3. Cross-encoder reranking

**Time Complexity**: O(Q × K + C × log C) where:
- Q: number of tokens in query
- K: average categories per token (~10-100)
- C: number of candidates (~10-100)

**Use Case**: Essential for large-scale datasets with 100K+ categories

**Minh họa cách hoạt động**:
```
Query: "sữa rửa mặt"
Categories: [
  "Điện thoại iPhone",          # index: 0
  "Sữa rửa mặt La Roche-Posay", # index: 1
  "Sữa tắm gội",               # index: 2
  "Mỹ phẩm chăm sóc da",       # index: 3
  "Sữa rửa mặt tạo bọt"        # index: 4
]

Bước 1 - Build Inverted Index:
- "điện": {0}
- "thoại": {0}
- "iphone": {0}
- "sữa": {1, 2, 4}     # các categories chứa token "sữa"
- "rửa": {1, 4}        # các categories chứa token "rửa"
- "mặt": {1, 4}        # các categories chứa token "mặt"
- "la": {1}
- "roche": {1}
- "posay": {1}
- "tắm": {2}
- "gội": {2}
- "mỹ": {3}
- "phẩm": {3}
- "chăm": {3}
- "sóc": {3}
- "da": {3}
- "tạo": {4}
- "bọt": {4}

Bước 2 - Query Processing:
- Query: "sữa rửa mặt" → tokens = {"sữa", "rửa", "mặt"}
- Tìm intersection: tokens["sữa"] ∩ tokens["rửa"] ∩ tokens["mặt"]
- = {1, 2, 4} ∩ {1, 4} ∩ {1, 4} = {1, 4}

Bước 3 - Lấy candidates:
- Các categories phù hợp: ["Sữa rửa mặt La Roche-Posay" (index: 1), "Sữa rửa mặt tạo bọt" (index: 4)]

Bước 4 - Cross-Encoder Reranking:
- So sánh: ["sữa rửa mặt", "Sữa rửa mặt La Roche-Posay"] và ["sữa rửa mặt", "Sữa rửa mặt tạo bọt"]
- Cross-encoder chọn ứng viên phù hợp nhất
- Kết quả: "Sữa rửa mặt tạo bọt" (gần với query hơn)
```

### 6. N-gram Index (`method_scalable_inverted_ngram_hybrid.py`)

**Approach**: Scalable method using n-gram matching for better partial matching

**Pipeline**:
1. Build n-gram index from all categories (unigrams, bigrams, trigrams)
2. Generate n-grams from query
3. Find matches and score candidates
4. Cross-encoder reranking

**Use Case**: When dealing with partial matches and typos in queries

**Minh họa cách hoạt động**:
```
Query: "iphone 13"
Categories: [
  "Điện thoại iPhone 13 Pro",  # index: 0
  "iPhone 13",                # index: 1
  "Phụ kiện iPhone",          # index: 2
  "Samsung Galaxy",           # index: 3
  "iPhone 13 Pro Max"         # index: 4
]

Bước 1 - Build N-gram Index:
Điện thoại iPhone 13 Pro (index 0):
  - unigrams: {"điện", "thoại", "iphone", "13", "pro"}
  - bigrams: {"điện thoại", "thoại iphone", "iphone 13", "13 pro"}
  - trigrams: {"điện thoại iphone", "thoại iphone 13", "iphone 13 pro"}
  → N-gram index: {"điện": {0}, "thoại": {0}, "iphone": {0,1,2}, "13": {0,1,4}, "pro": {0,4}, "điện thoại": {0}, "thoại iphone": {0}, "iphone 13": {0}, "13 pro": {0}, ...}

iPhone 13 (index 1):
  - unigrams: {"iphone", "13"}
  - bigrams: {"iphone 13"}
  - trigrams: {} (không đủ 3 token)
  → Thêm vào index: {"iphone": {0,1,2}, "13": {0,1,4}, "iphone 13": {1}, ...}

Phụ kiện iPhone (index 2):
  - unigrams: {"phụ", "kiện", "iphone"}
  - bigrams: {"phụ kiện", "kiện iphone"}
  - trigrams: {}
  → Thêm vào index: {"phụ": {2}, "kiện": {2}, "iphone": {0,1,2}, "phụ kiện": {2}, "kiện iphone": {2}, ...}

Bước 2 - Query N-gram Generation:
- Query: "iphone 13" → tokens = ["iphone", "13"]
- Unigrams: ["iphone", "13"] → scores: [1.0, 1.0]
- Bigrams: ["iphone 13"] → score: [2.0] (ưu tiên bigram dài hơn)

Bước 3 - Tìm candidates từ n-gram index:
- "iphone" → {0, 1, 2} (Điện thoại iPhone 13 Pro, iPhone 13, Phụ kiện iPhone)
- "13" → {0, 1, 4} (Điện thoại iPhone 13 Pro, iPhone 13, iPhone 13 Pro Max)
- "iphone 13" → {0, 1} (Điện thoại iPhone 13 Pro, iPhone 13)
- Gộp và tính score:
  - Category 0 ("Điện thoại iPhone 13 Pro"): score = 2.0 (bigram) + 1.0 (iphone) + 1.0 (13) = 4.0
  - Category 1 ("iPhone 13"): score = 2.0 (bigram) + 1.0 (iphone) + 1.0 (13) = 4.0
  - Category 2 ("Phụ kiện iPhone"): score = 1.0 (iphone) = 1.0
  - Category 4 ("iPhone 13 Pro Max"): score = 1.0 (13) + 1.0 (iphone) = 2.0

Bước 4 - Cross-Encoder Reranking:
- So sánh các candidates với query qua cross-encoder
- ["iphone 13", "iPhone 13"] → score cao nhất do match chính xác
- Kết quả: "iPhone 13"
```

### 7. Hybrid (`method_scalable_inverted_ngram_hybrid.py`)

**Approach**: Combines inverted index (lexical) with semantic search

**Pipeline**:
1. Inverted index retrieval → ~20-30 lexical candidates
2. Semantic search → ~20-30 semantic candidates
3. Union to get ~40-60 unique candidates
4. Cross-encoder reranking

**Use Case**: Best of both worlds - handles exact matches and semantic similarity

**Minh họa cách hoạt động**:
```
Query: "smartphone"
Categories: [
  "Điện thoại thông minh",      # index: 0
  "Mobile",                      # index: 1
  "iPhone",                      # index: 2
  "Samsung",                     # index: 3
  "Máy tính bảng",              # index: 4
  "Laptop",                      # index: 5
  "Điện thoại di động"          # index: 6
]

Bước 1 - Inverted Index Retrieval:
- Query "smartphone" → normalize → tokens = {"smartphone"} (không match trực tiếp)
- Không có match chính xác, fallback: Union của các token gần giống
- Giả sử hệ thống có fuzzy matching tìm được: {"phone", "mobile"} (nếu có)
- Hoặc nếu query được dịch: "smartphone" → "điện thoại thông minh" → tokens = {"điện", "thoại", "thông", "minh"}
  → Match với các category chứa các từ này
- Lexical candidates: ["Điện thoại thông minh" (0), "Mobile" (1), "Điện thoại di động" (6)]

Bước 2 - Semantic Retrieval:
- Encode query "smartphone" thành vector A
- Encode tất cả categories thành vectors [B0, B1, B2, B3, B4, B5, B6]
- Tính cosine similarity: cos(A, Bi) với i=0..6
- Top semantic matches (giả sử):
  - "Điện thoại thông minh" (sim=0.9) - index: 0
  - "Điện thoại di động" (sim=0.85) - index: 6
  - "Mobile" (sim=0.78) - index: 1
  - "iPhone" (sim=0.65) - index: 2

Bước 3 - Union Candidates:
- Lexical: {0, 1, 6} ("Điện thoại thông minh", "Mobile", "Điện thoại di động")
- Semantic: {0, 6, 1, 2} (top 4)
- Union: {0, 1, 6, 2} = ["Điện thoại thông minh", "Mobile", "Điện thoại di động", "iPhone"]

Bước 4 - Cross-Encoder Reranking:
- So sánh từng cặp query-candidate:
  - ["smartphone", "Điện thoại thông minh"] → score=0.95
  - ["smartphone", "Mobile"] → score=0.87
  - ["smartphone", "Điện thoại di động"] → score=0.83
  - ["smartphone", "iPhone"] → score=0.75
- Top score: "Điện thoại thông minh"
- Kết quả: "Điện thoại thông minh" (match cả lexical và semantic)
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Query Input   │───▶│  Method Selector │───▶│  Method Pipeline│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
           ┌──────────────────────────────────────────────┘
           │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Priority Calc.  │───▶│  Retrieval       │───▶│ Cross-Encoder   │
│ (Specificity,   │    │  (BM25, Prefix,  │    │  Reranking      │
│  Overlap, Len)  │    │   Inverted, etc) │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

- **Priority Calculator**: Calculates composite scores based on specificity (IDF-based), token overlap with query, and category length
- **Retrieval Models**: Different algorithms for finding candidate categories
- **Cross-Encoder**: Final reranking using semantic similarity between query and category
- **Scalable Indexes**: Optimized data structures for handling 100K+ categories

## Usage Examples

### Single Product Search

```python
from methods.method_bm25_lexical_cross import method_optimized_pipeline
from models.cross_encoder import Qwen3CrossEncoder

# Initialize models
cross_encoder = Qwen3CrossEncoder()

query = "iPhone 13 Pro Max 256GB"
categories = [
    "Điện thoại",
    "iPhone", 
    "iPhone 13",
    "iPhone 13 Pro Max",
    "Điện tử tiêu dùng",
    "Thiết bị di động"
]

result, score, method_used, debug_info = method_optimized_pipeline(
    query=query,
    categories=categories,
    cross_encoder=cross_encoder,
    semantic_encoder=None,
    idx=0
)

print(f"Best match: {result}")
print(f"Confidence score: {score:.3f}")
print(f"Method used: {method_used}")
print(f"Debug info: {debug_info}")
```

### Batch Evaluation

The system also includes a batch evaluation feature that processes CSV files:

```python
# CSV format:
# tên hàng hóa,kết quả mong muốn,danh mục
# iPhone 13 Pro Max,Điện thoại iPhone,"Điện thoại,Điện tử,iPhone,iPhone 13,iPhone 13 Pro Max,Điện tử tiêu dùng"
```

## Performance Comparison

| Method | Accuracy | Use Case | Complexity |
|--------|----------|----------|------------|
| Adaptive Lexical | Good | Small datasets, prefix matching | O(n) |
| Adaptive Semantic | Good | Semantic similarity important | O(n×d) |
| BM25 Lexical | Very Good | Varied query lengths | O(n) |
| Optimized BM25 | Excellent | Best lexical approach | O(n) |
| Inverted Index | Good | 100K+ categories | O(1) |
| N-gram Index | Good | Partial matching | O(1) |
| Hybrid | Best | Best overall performance | O(1) + O(n×d) |

Where n is the number of categories and d is the embedding dimension.

## Installation

1. Clone the repository
2. Install dependencies (see requirements if available)
3. Run the application:

```bash
python app.py
```

The application provides both a single search interface and batch evaluation capabilities.