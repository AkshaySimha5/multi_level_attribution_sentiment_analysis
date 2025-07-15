import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr import LayerConductance
from captum.metrics import infidelity
from captum.attr import visualization as viz
import logging
from utils.preprocess import TextPreprocessor
from pathlib import Path

logger = logging.getLogger(__name__)

class LayerConductanceAnalyzer:
    def __init__(self, model, tokenizer, label_names, device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.label_names = label_names
        self.device = device
        self.preprocessor = TextPreprocessor(tokenizer, device)

        self.layer_dict = {
            "embeddings": self.model.distilbert.embeddings,
            "layer_0": self.model.distilbert.transformer.layer[0],
            "layer_2": self.model.distilbert.transformer.layer[2],
            "layer_4": self.model.distilbert.transformer.layer[4],
        }

    def forward_func(self, input_ids, attention_mask):
        """Standard forward function using input_ids"""
        input_ids = input_ids.long()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return F.softmax(outputs.logits, dim=-1)

    def forward_embeds(self, inputs_embeds, attention_mask):
        """Forward function that accepts embeddings directly"""
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return F.softmax(outputs.logits, dim=-1)

    def get_attributions(self, text, target_label):
        """Get Layer Conductance attributions for all specified layers"""
        inputs = self.preprocessor.preprocess(text)
        input_ids = inputs["input_ids"].long()
        attention_mask = inputs["attention_mask"]

        results = {}

        for layer_name, layer in self.layer_dict.items():
            lc = LayerConductance(self.forward_func, layer)
            attributions = lc.attribute(
                inputs=input_ids,
                additional_forward_args=(attention_mask,),
                target=target_label
            )
            
            # Sum across hidden dimension and normalize
            attr_sum = attributions.sum(dim=-1).squeeze(0)
            attr_norm = attr_sum / torch.norm(attr_sum)

            results[layer_name] = {
                "attributions": attributions,  # Keep original for infidelity
                "normalized_attributions": attr_norm
            }

        return results, input_ids, attention_mask

    def save_matplotlib_plot(self, results, input_ids, target_label, save_path=None):
        """Save matplotlib bar chart visualization"""
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Filter out special tokens
        filtered_tokens = []
        filtered_indices = []
        for i, token in enumerate(tokens):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                filtered_tokens.append(token)
                filtered_indices.append(i)

        plt.figure(figsize=(14, 8))
        
        for i, (layer_name, result) in enumerate(results.items()):
            plt.subplot(len(results), 1, i+1)
            attributions = result["normalized_attributions"]
            
            # Filter attributions for non-special tokens
            filtered_attrs = [attributions[i].item() for i in filtered_indices]

            plt.bar(range(len(filtered_tokens)), filtered_attrs)
            plt.xticks(range(len(filtered_tokens)), filtered_tokens, rotation=45, ha='right')
            plt.title(f"Layer Conductance – {layer_name} (Target: {self.label_names[target_label]})")
            plt.ylabel("Normalized Attribution")
            
        plt.xlabel("Token Position")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Layer Conductance bar plot to {save_path}")
        plt.close()

    def save_layerwise_comparison_plot(self, results, input_ids, target_label, save_path=None):
        """Save line graph showing normalized attributions across layers"""
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Filter out special tokens
        filtered_tokens = []
        filtered_indices = []
        for i, token in enumerate(tokens):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                filtered_tokens.append(token)
                filtered_indices.append(i)

        # Define colors for each layer
        colors = {
            "embeddings": "#1f77b4",  # Blue
            "layer_0": "#ff7f0e",     # Orange
            "layer_2": "#2ca02c",     # Green
            "layer_4": "#d62728"      # Red
        }

        plt.figure(figsize=(15, 8))

        for layer_name, result in results.items():
            attributions = result["normalized_attributions"]
            
            # Filter attributions for non-special tokens
            filtered_attrs = [attributions[i].item() for i in filtered_indices]

            plt.plot(range(len(filtered_tokens)), filtered_attrs,
                    color=colors[layer_name], marker='o', linewidth=2,
                    markersize=4, label=layer_name, alpha=0.8)

        plt.xlabel('Words', fontsize=12)
        plt.ylabel('Normalized Attribution', fontsize=12)
        plt.title(f'Layer Conductance - Layer-wise Attribution Comparison - Target: {self.label_names[target_label]}', fontsize=14)
        plt.xticks(range(len(filtered_tokens)), filtered_tokens, rotation=45, ha='right')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Layer Conductance comparison plot to {save_path}")
        plt.close()

    def save_captum_html(self, vis_records, save_path="output/lc_visualization.html"):
        """Save Captum HTML visualization"""
        html_obj = viz.visualize_text(vis_records)
        html_str = html_obj.data if hasattr(html_obj, "data") else str(html_obj)
        Path(save_path).write_text(html_str, encoding="utf-8")
        logger.info(f"Saved Captum visualization as HTML to {save_path}")

    def visualize_multiple_layers(self, results, input_ids, attention_mask, target_label, html_path=None):
        """Create Captum visualization records for all layers"""
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        probs = self.forward_func(input_ids, attention_mask)
        pred_label = torch.argmax(probs, dim=1).item()
        pred_score = torch.max(probs).item()

        vis_records = []

        for layer_name, result in results.items():
            attributions = result["normalized_attributions"]
            
            # Filter out special tokens
            filtered_tokens = []
            filtered_attrs = []
            for token, attr in zip(tokens, attributions):
                if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                    filtered_tokens.append(token)
                    filtered_attrs.append(attr.item())

            vis_records.append(
                viz.VisualizationDataRecord(
                    filtered_attrs,
                    pred_score,
                    f"{self.label_names[pred_label]} ({layer_name})",
                    self.label_names[target_label],
                    self.label_names[target_label],
                    sum(filtered_attrs),
                    filtered_tokens,
                    0.0  # No convergence delta for Layer Conductance
                )
            )

        if html_path:
            self.save_captum_html(vis_records, html_path)

    def compute_infidelity(self, text, target_label, noise_level=0.1, n_perturb_samples=20):
        """
        Compute infidelity metric on embeddings layer
        
        Args:
            text: Input text
            target_label: Target class label
            noise_level: Standard deviation of noise for perturbations
            n_perturb_samples: Number of perturbation samples for infidelity computation
        """
        inputs = self.preprocessor.preprocess(text)
        input_ids = inputs["input_ids"].long()
        attention_mask = inputs["attention_mask"]

        # Get Layer Conductance attributions on input_ids (using the embeddings layer)
        lc = LayerConductance(self.forward_func, self.model.distilbert.embeddings)
        attributions = lc.attribute(
            inputs=input_ids,
            additional_forward_args=(attention_mask,),
            target=target_label
        )

        # Convert input_ids to embeddings for infidelity computation
        embedding_layer = self.model.distilbert.embeddings
        embeds = embedding_layer(input_ids)

        # Define perturbation function
        def perturb_fn(x):
            """Add Gaussian noise to embeddings"""
            noise = torch.randn_like(x) * noise_level
            return noise, x + noise

        # Compute infidelity
        infid = infidelity(
            forward_func=self.forward_embeds,
            perturb_func=perturb_fn,
            inputs=embeds,
            attributions=attributions,
            additional_forward_args=(attention_mask,),
            target=target_label,
            n_perturb_samples=n_perturb_samples
        )

        logger.info(f"Infidelity: {infid.item():.6f}, Noise Level: {noise_level}, Perturbation Samples: {n_perturb_samples}")
        return infid.item()

    def analyze(self, text, target_label, save_viz=False, viz_path=None, comparison_path=None):
        """Run complete Layer Conductance attribution analysis"""
        logger.info(f"Analyzing text for target: {self.label_names[target_label]}")

        # Get layerwise attributions
        results, input_ids, attention_mask = self.get_attributions(text, target_label)

        if save_viz:
            # Save bar chart visualization
            if viz_path:
                self.save_matplotlib_plot(results, input_ids, target_label, save_path=viz_path)
            
            # Save layerwise comparison plot
            if comparison_path:
                self.save_layerwise_comparison_plot(results, input_ids, target_label, save_path=comparison_path)
            
            # Save HTML visualization
            self.visualize_multiple_layers(results, input_ids, attention_mask, target_label,
                                         html_path="output/lc_visualization.html")

        # Compute infidelity on embeddings
        infid_score = self.compute_infidelity(text, target_label)

        return {
            "layer_attributions": results,
            "infidelity_score": infid_score,
            "text": text,
            "target_label": target_label
        }