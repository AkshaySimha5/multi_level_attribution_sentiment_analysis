import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr import LayerIntegratedGradients
from captum.metrics import infidelity
from captum.attr import visualization as viz
import logging
from utils.preprocess import TextPreprocessor
from pathlib import Path

logger = logging.getLogger(__name__)

class LayerIntegratedGradientsAnalyzer:
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
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return F.softmax(outputs.logits, dim=-1)

    def forward_embeds(self, inputs_embeds, attention_mask):
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return F.softmax(outputs.logits, dim=-1)

    def get_attributions(self, text, target_label):
        inputs = self.preprocessor.preprocess(text)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)

        results = {}
        for layer_name, layer in self.layer_dict.items():
            lig = LayerIntegratedGradients(self.forward_func, layer)
            attributions, delta = lig.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=(attention_mask,),
                target=target_label,
                return_convergence_delta=True
            )
            results[layer_name] = {
                "attributions": attributions,
                "delta": delta.item()
            }

        return results, input_ids, attention_mask

    def save_matplotlib_plot(self, results, input_ids, target_label, save_path=None):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        filtered_tokens = []
        filtered_indices = []
        for i, token in enumerate(tokens):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                filtered_tokens.append(token)
                filtered_indices.append(i)

        colors = {
            "embeddings": "#1f77b4",
            "layer_0": "#ff7f0e",
            "layer_2": "#2ca02c",
            "layer_4": "#d62728"
        }

        plt.figure(figsize=(15, 8))

        for layer_name, result in results.items():
            attr = result["attributions"].sum(dim=-1).squeeze(0)
            norm_attr = attr / torch.norm(attr)
            filtered_attrs = [norm_attr[i].item() for i in filtered_indices]

            plt.plot(range(len(filtered_tokens)), filtered_attrs,
                     color=colors[layer_name], marker='o', linewidth=2,
                     markersize=4, label=layer_name, alpha=0.8)

        plt.xlabel('Words')
        plt.ylabel('Normalized Attribution')
        plt.title(f'Layer-wise Attribution - Target: {self.label_names[target_label]}')
        plt.xticks(range(len(filtered_tokens)), filtered_tokens, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved LIG plot to {save_path}")
        plt.close()

    def save_captum_html(self, vis_records, save_path="output/lig_visualization.html"):
        html_obj = viz.visualize_text(vis_records)
        html_str = html_obj.data if hasattr(html_obj, "data") else str(html_obj)
        Path(save_path).write_text(html_str, encoding="utf-8")
        logger.info(f"Saved Captum visualization as HTML to {save_path}")

    def visualize_multiple_layers(self, results, input_ids, attention_mask, target_label, html_path=None):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        probs = self.forward_func(input_ids, attention_mask)
        pred_label = torch.argmax(probs, dim=1).item()
        pred_score = torch.max(probs).item()

        vis_records = []

        for layer_name, result in results.items():
            attributions = result["attributions"].sum(dim=-1).squeeze(0)
            attr_norm = attributions / torch.norm(attributions)

            filtered_tokens = []
            filtered_attrs = []
            for token, attr in zip(tokens, attr_norm):
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
                    result["delta"]
                )
            )

        if html_path:
            self.save_captum_html(vis_records, html_path)

    def compute_infidelity(self, text, target_label, noise_level=0.1, n_perturb_samples=20):
        inputs = self.preprocessor.preprocess(text)
        input_ids = inputs["input_ids"].long()
        attention_mask = inputs["attention_mask"]
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id).long()

        embedding_layer = self.model.distilbert.embeddings
        embeds = embedding_layer(input_ids)
        baseline_embeds = embedding_layer(baseline_ids)

        lig = LayerIntegratedGradients(self.forward_embeds, self.model.distilbert.transformer.layer[0])
        attributions, delta = lig.attribute(
            inputs=embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask,),
            target=target_label,
            return_convergence_delta=True
        )

        def perturb_fn(x):
            noise = torch.randn_like(x) * noise_level
            return noise, x + noise

        infid = infidelity(
            forward_func=self.forward_embeds,
            perturb_func=perturb_fn,
            inputs=embeds,
            attributions=attributions,
            additional_forward_args=(attention_mask,),
            target=target_label,
            n_perturb_samples=n_perturb_samples
        )

        logger.info(f"Infidelity: {infid.item():.6f}, Convergence Delta: {delta.item():.6f}")
        return infid.item(), delta.item()

    def analyze(self, text, target_label, save_viz=False, viz_path=None):
        logger.info(f"Analyzing text for target: {self.label_names[target_label]}")

        results, input_ids, attention_mask = self.get_attributions(text, target_label)

        if save_viz:
            self.save_matplotlib_plot(results, input_ids, target_label, save_path=viz_path)
            self.visualize_multiple_layers(results, input_ids, attention_mask, target_label,
                                           html_path="output/lig_visualization.html")

        infid_score, delta_score = self.compute_infidelity(text, target_label)

        return {
            "layer_attributions": results,
            "infidelity_score": infid_score,
            "convergence_delta": delta_score,
            "text": text,
            "target_label": target_label
        }
