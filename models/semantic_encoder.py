from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np


class SemanticEncoder:
    """Encoder để tạo embeddings cho semantic search"""

    def __init__(
        self, model_name: str = "llm/Qwen3-Embedding-0.6B", device: str = None
    ):
        print("Loading Semantic Encoder...")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        print(f"Semantic Encoder loaded on {device}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts thành embeddings"""
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()
