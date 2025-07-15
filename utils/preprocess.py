import torch

class TextPreprocessor:
    def __init__(self, tokenizer, device, max_length=512):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
    
    def preprocess(self, text):
        """Tokenize and prepare text for model input"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def tokens_to_string(self, input_ids):
        """Convert token IDs back to readable tokens"""
        return self.tokenizer.convert_ids_to_tokens(input_ids[0])
