import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from captum.metrics import infidelity
from captum.attr import visualization as viz
import logging
from utils.preprocess import TextPreprocessor

logger = logging.getLogger(__name__)

class IntegratedGradientsAnalyzer:
    def __init__(self, model, tokenizer, label_names, device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.label_names = label_names
        self.device = device
        self.preprocessor = TextPreprocessor(tokenizer, device)

    def forward_func(self, input_ids, attention_mask):
        """Forward function for token-level attributions"""
        input_ids = input_ids.long()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return F.softmax(outputs.logits, dim=-1)

    def forward_embeds(self, embeds, attention_mask):
        """Forward function for embedding-level attributions"""
        outputs = self.model.distilbert(inputs_embeds=embeds, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_output = hidden_states[:, 0]
        logits = self.model.classifier(cls_output)
        return torch.nn.functional.softmax(logits, dim=-1)

    def get_attributions(self, text, target_label):
        """Get integrated gradients attributions for given text"""
        inputs = self.preprocessor.preprocess(text)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        embedding_layer = self.model.distilbert.embeddings
        embeds = embedding_layer(input_ids)

        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        baseline_embeds = embedding_layer(baseline_ids)

        ig = IntegratedGradients(self.forward_embeds)
        attributions, delta = ig.attribute(
            inputs=embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask,),
            target=target_label,
            return_convergence_delta=True
        )

        return {
            "attributions": attributions,
            "delta": delta,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "embeds": embeds,
            "baseline_embeds": baseline_embeds
        }

    def generate_visualization_data(self, result, target_label):
        """Generate visualization data without actually displaying"""
        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]
        attributions = result["attributions"]
        delta = result["delta"]

        tokens = self.preprocessor.tokens_to_string(input_ids)
        probs = self.forward_func(input_ids, attention_mask)
        pred_label = torch.argmax(probs, dim=1).item()
        pred_score = torch.max(probs).item()

        attr_sum = attributions.sum(dim=-1).squeeze(0)
        attr_norm = attr_sum / torch.norm(attr_sum)

        # Filter out special tokens
        filtered_tokens = []
        filtered_attrs = []
        for token, attr in zip(tokens, attr_norm):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                filtered_tokens.append(token)
                filtered_attrs.append(attr.item())

        return {
            "tokens": filtered_tokens,
            "attributions": filtered_attrs,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "delta": delta.item(),
            "target_label": target_label
        }

    def save_visualization(self, viz_data, save_path=None):
        """Save visualization to file instead of showing"""
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(viz_data["tokens"])), viz_data["attributions"])
        plt.xticks(range(len(viz_data["tokens"])), viz_data["tokens"], rotation=45)
        plt.title(f"Integrated Gradients (Target: {self.label_names[viz_data['target_label']]})")
        plt.ylabel("Normalized Attribution")
        plt.xlabel("Token Position")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        return plt

    def compute_infidelity(self, result, target_label, noise_level=0.1, n_perturb_samples=20):
        """Compute infidelity score for attribution quality"""
        embeds = result["embeds"]
        attention_mask = result["attention_mask"]
        attributions = result["attributions"]

        def perturb_fn(input_embeds):
            noise = torch.randn_like(input_embeds) * noise_level
            return noise, input_embeds + noise

        infid = infidelity(
            forward_func=self.forward_embeds,
            perturb_func=perturb_fn,
            inputs=embeds,
            attributions=attributions,
            additional_forward_args=(attention_mask,),
            target=target_label,
            n_perturb_samples=n_perturb_samples
        )

        logger.info(f"Infidelity score: {infid.item():.6f}")
        return infid.item()

    def analyze(self, text, target_label, save_viz=False, viz_path=None):
        """Main analysis method - returns data instead of printing"""
        logger.info(f"Analyzing text for target: {self.label_names[target_label]}")
        
        # Get attributions
        result = self.get_attributions(text, target_label)
        
        # Generate visualization data
        viz_data = self.generate_visualization_data(result, target_label)
        
        # Compute infidelity
        infidelity_score = self.compute_infidelity(result, target_label)
        
        # Save visualization if requested
        if save_viz:
            self.save_visualization(viz_data, viz_path)
        
        return {
            "attributions": result,
            "visualization": viz_data,
            "infidelity_score": infidelity_score,
            "text": text,
            "target_label": target_label
        }
