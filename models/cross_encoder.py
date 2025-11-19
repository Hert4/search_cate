from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


class Qwen3CrossEncoder:
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B", device: str = None):
        print("Loading Qwen3-Reranker...")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        
        self.prefix = "\n\n<system>\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".\n\n<user>\n"
        self.suffix = "\n\n<assistant>\n\n\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        self.task_instruction = "Given a product search query, judge whether the product matches the user's search intent"
        print(f"✅ Cross-encoder loaded on {device}")
    
    def format_instruction(self, query: str, doc: str) -> str:
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=self.task_instruction, query=query, doc=doc
        )
    
    def process_inputs(self, pairs: List[str]):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, 
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        return inputs
    
    def compute_logits(self, inputs):
        with torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def predict(self, pairs: List[List[str]]) -> List[float]:
        formatted_pairs = [self.format_instruction(query, doc) for query, doc in pairs]
        inputs = self.process_inputs(formatted_pairs)
        scores = self.compute_logits(inputs)
        return scores