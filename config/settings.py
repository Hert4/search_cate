"""
Configuration settings for the search system
"""

# Model settings
SEMANTIC_ENCODER_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CROSS_ENCODER_MODEL = "Qwen/Qwen3-Reranker-0.6B"

# Search settings
TOP_N_RESULTS = 50
MAX_LENGTH = 8192

# Device settings
DEVICE = "cuda"  # Change to "cpu" if CUDA is not available

# File paths
DEFAULT_DATA_PATH = "/data/small-language-models/duc/Searching_cate/data/data.csv"