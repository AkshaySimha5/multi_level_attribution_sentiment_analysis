import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

class SentimentModel:
    def __init__(self, model_path=None, tokenizer_name="distilbert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path and os.path.exists(model_path):
            # Load your fine-tuned model
            self.model = torch.load(model_path, map_location=self.device)
        else:
            # Fallback to pre-trained model
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model.to(self.device).eval()
        self.label_names = ["Negative", "Positive"]

    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_device(self):
        return self.device