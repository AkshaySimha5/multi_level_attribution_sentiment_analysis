import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import NeuronConductance
from captum.attr import visualization as viz
import os
import logging
from utils.preprocess import TextPreprocessor
from captum.attr._utils.visualization import VisualizationDataRecord

logger = logging.getLogger(__name__)

class NeuronConductanceAnalyzer:
    def __init__(self, model, tokenizer, label_names, device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.label_names = label_names
        self.device = device
        self.preprocessor = TextPreprocessor(tokenizer, device)

        assert isinstance(self.model.pre_classifier, torch.nn.Module), \
            "Model must have a pre_classifier layer for NeuronConductance"

    def forward_func(self, embeds, attention_mask):
        embeds.requires_grad_()
        outputs = self.model.distilbert(inputs_embeds=embeds, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        hidden = self.model.pre_classifier(cls_output)
        hidden = F.relu(hidden)
        logits = self.model.classifier(hidden)
        return logits

    def get_attributions(self, text, target_label, neuron_idx=50):
        inputs = self.preprocessor.preprocess(text)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        embeds = self.model.distilbert.embeddings(input_ids).detach().clone().requires_grad_()

        nc = NeuronConductance(self.forward_func, self.model.pre_classifier)
        attributions = nc.attribute(
            inputs=embeds,
            neuron_selector=neuron_idx,
            additional_forward_args=(attention_mask,),
            target=target_label,
            n_steps=25,
            internal_batch_size=1
        )

        return {
            "attributions": attributions,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "embeds": embeds,
            "neuron_idx": neuron_idx
        }

    def generate_visualization_data(self, result, target_label):
        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]
        attributions = result["attributions"]
        neuron_idx = result["neuron_idx"]

        tokens = self.preprocessor.tokens_to_string(input_ids)
        probs = F.softmax(self.model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        pred_score = torch.max(probs).item()

        attr_sum = attributions.sum(dim=-1).squeeze(0)
        attr_norm = attr_sum / torch.norm(attr_sum)

        filtered_tokens, filtered_attrs = [], []
        for token, attr in zip(tokens, attr_norm):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                filtered_tokens.append(token)
                filtered_attrs.append(attr.item())

        return {
            "tokens": filtered_tokens,
            "attributions": filtered_attrs,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "target_label": target_label,
            "neuron_idx": neuron_idx
        }

    def compute_activations(self, input_ids, attention_mask, neuron_idx):
        embeds = self.model.distilbert.embeddings(input_ids)
        seq_len = input_ids.size(1)
        activations = []

        for pos in range(seq_len):
            masked = torch.zeros_like(embeds)
            masked[:, pos, :] = embeds[:, pos, :]
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=embeds.dtype)
            extended_mask = (1.0 - extended_mask) * -10000.0

            out = self.model.distilbert.transformer(
                masked,
                extended_mask,
                head_mask=[None] * len(self.model.distilbert.transformer.layer),
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )

            pooled = out.last_hidden_state[:, 0, :]
            hidden = self.model.pre_classifier(pooled)
            activations.append(hidden.squeeze(0)[neuron_idx].item())

        return activations

    def save_visualizations(self, viz_data, activations, save_dir="output", base_name="neuron_conductance"):
        os.makedirs(save_dir, exist_ok=True)
        tokens, scores = viz_data["tokens"], viz_data["attributions"]
        colors = ["green" if s > 0 else "red" for s in scores]
        positions = range(len(tokens))

        # Filter activations to match token filtering
        filtered_activations = activations[:len(tokens)]
        assert len(filtered_activations) == len(tokens), "Mismatch between activations and tokens."

        # Bar plot
        bar_path = os.path.join(save_dir, f"{base_name}_bar.png")
        plt.figure(figsize=(12, 6))
        plt.bar(positions, scores, color=colors)
        plt.xticks(positions, tokens, rotation=45)
        plt.title(f"Attribution Scores for Neuron {viz_data['neuron_idx']}")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(bar_path, dpi=300)
        plt.close()

        # Attribution vs Activation
        actcomp_path = os.path.join(save_dir, f"{base_name}_activation_vs_attr.png")
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.bar(positions, scores, color=colors)
        plt.title("Attribution Scores")
        plt.xticks(positions, tokens, rotation=45)

        plt.subplot(1, 2, 2)
        plt.plot(positions, filtered_activations, marker='o', color='blue')
        plt.title("Neuron Activation")
        plt.xticks(positions, tokens, rotation=45)
        plt.tight_layout()
        plt.savefig(actcomp_path, dpi=300)
        plt.close()

        # Heatmap
        heatmap_path = os.path.join(save_dir, f"{base_name}_heatmap.png")
        plt.figure(figsize=(10, 1.5))
        sns.heatmap(np.array([scores]), annot=True, cmap="RdYlGn", center=0,
                    xticklabels=tokens, yticklabels=["Attr"], fmt=".2f", linewidths=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()

        # Generate HTML visualization
        html_path = os.path.join(save_dir, f"{base_name}.html")
        try:
            # Create a VisualizationDataRecord with correct parameter name
            vis_record = VisualizationDataRecord(
                word_attributions=scores,
                pred_prob=viz_data["pred_score"],
                pred_class=self.label_names[viz_data["pred_label"]],
                true_class=self.label_names[viz_data["target_label"]],
                attr_class=self.label_names[viz_data["target_label"]],
                attr_score=sum(scores),
                raw_input_ids=" ".join(tokens),  # Changed from raw_input to raw_input_ids
                convergence_score=0.0  # Changed from None to 0.0
            )

            # Generate HTML
            html_out = viz.visualize_text([vis_record])
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_out.data)
            logger.info(f"Saved HTML: {html_path}")
        except Exception as e:
            logger.warning(f"Failed to generate HTML visualization: {e}")
            # Create a simple HTML fallback
            html_content = f"""
            <html>
            <head><title>Neuron Conductance Analysis</title></head>
            <body>
                <h1>Neuron Conductance Analysis - Neuron {viz_data['neuron_idx']}</h1>
                <p><strong>Text:</strong> {' '.join(tokens)}</p>
                <p><strong>Predicted Class:</strong> {self.label_names[viz_data['pred_label']]} ({viz_data['pred_score']:.3f})</p>
                <p><strong>Target Class:</strong> {self.label_names[viz_data['target_label']]}</p>
                <h2>Token Attributions:</h2>
                <table border="1">
                    <tr><th>Token</th><th>Attribution</th></tr>
                    {''.join(f'<tr><td>{token}</td><td>{attr:.4f}</td></tr>' for token, attr in zip(tokens, scores))}
                </table>
            </body>
            </html>
            """
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Saved fallback HTML: {html_path}")

        logger.info(f"Saved bar: {bar_path}")
        logger.info(f"Saved activation vs attr: {actcomp_path}")
        logger.info(f"Saved heatmap: {heatmap_path}")

        return {
            "bar": bar_path,
            "activation_comparison": actcomp_path,
            "heatmap": heatmap_path,
            "html": html_path
        }

    def analyze(self, text, target_label, neuron_idx=50,
                save_viz=True, save_dir="output", base_name="neuron_conductance"):
        logger.info(f"Running Neuron Conductance for neuron {neuron_idx} on: {text}")
        result = self.get_attributions(text, target_label, neuron_idx=neuron_idx)
        viz_data = self.generate_visualization_data(result, target_label)
        activations = self.compute_activations(result["input_ids"], result["attention_mask"], neuron_idx)

        saved_paths = {}
        if save_viz:
            saved_paths = self.save_visualizations(viz_data, activations, save_dir, base_name)

        return {
            "attributions": result,
            "visualization": viz_data,
            "text": text,
            "target_label": target_label,
            "neuron_idx": neuron_idx,
            "activation_per_token": activations,
            "infidelity_score": None,
            "visualization_paths": saved_paths
        }