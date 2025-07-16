import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr import KernelShap, visualization as viz
from captum.metrics import infidelity
from utils.preprocess import TextPreprocessor
import logging

logger = logging.getLogger(__name__)

class KernelShapAnalyzer:
    def __init__(self, model, tokenizer, label_names, device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.label_names = label_names
        self.device = device
        self.preprocessor = TextPreprocessor(tokenizer, device)

        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id
        self.mask_id = tokenizer.mask_token_id

    def _masked_forward(self, feat_ids, mask):
        B = mask.size(0)
        masked = feat_ids.unsqueeze(0).repeat(B, 1).clone()
        masked[mask == 0] = self.mask_id

        cls = torch.full((B, 1), self.cls_id, dtype=torch.long).to(self.device)
        sep = torch.full((B, 1), self.sep_id, dtype=torch.long).to(self.device)
        ids = torch.cat([cls, masked, sep], dim=1)
        att = torch.ones_like(ids)

        logits = self.model(ids, attention_mask=att).logits
        return F.softmax(logits, dim=-1)

    def get_attributions(self, text):
        enc = self.preprocessor.preprocess(text)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        full_ids = input_ids[0]
        sep_pos = (full_ids == self.sep_id).nonzero(as_tuple=True)[0][0].item()
        feat_ids = full_ids[1:sep_pos]
        tokens = self.tokenizer.convert_ids_to_tokens(feat_ids)

        probs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(probs.logits, dim=-1)
        pred_cls = int(torch.argmax(probs))
        pred_prob = float(probs[0, pred_cls])

        forward_fn = lambda m: self._masked_forward(feat_ids.to(self.device), m.to(self.device))[:, pred_cls]

        ks = KernelShap(forward_fn)
        baseline = torch.zeros(1, len(tokens)).to(self.device)
        inputs = torch.ones(1, len(tokens)).to(self.device)

        attr = ks.attribute(inputs, baselines=baseline, n_samples=100)[0].detach().cpu().numpy().copy()

        return {
            "attributions": attr,
            "tokens": tokens,
            "inputs_mask": inputs,
            "forward_fn": forward_fn,
            "pred_cls": pred_cls,
            "pred_prob": pred_prob
        }

    def generate_visualization_data(self, info):
        attr = np.array(info["attributions"], copy=True)
        attr_norm = attr / (np.linalg.norm(attr) + 1e-10)

        filtered_tokens, filtered_attrs = [], []
        for token, value in zip(info["tokens"], attr_norm):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                filtered_tokens.append(token)
                filtered_attrs.append(value)

        return {
            "tokens": filtered_tokens,
            "attributions": filtered_attrs,
            "pred_label": info["pred_cls"],
            "pred_score": info["pred_prob"]
        }

    def save_visualization(self, viz_data, png_path=None, html_path=None):
        # PNG Plot with green/red bars
        colors = ["green" if val > 0 else "red" for val in viz_data["attributions"]]
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(viz_data["tokens"])), viz_data["attributions"], color=colors)
        plt.xticks(range(len(viz_data["tokens"])), viz_data["tokens"], rotation=45)
        plt.title(f"Kernel SHAP (Predicted: {self.label_names[viz_data['pred_label']]})")
        plt.ylabel("Normalized Attribution")
        plt.xlabel("Token Position")
        plt.tight_layout()

        if png_path:
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            logger.info(f"Kernel SHAP bar plot saved to {png_path}")

        # Captum HTML Visualization
        if html_path:
            try:
                html = viz.visualize_text([
                viz.VisualizationDataRecord(
                    word_attributions=viz_data["attributions"],
                    pred_prob=viz_data["pred_score"],
                    pred_class=self.label_names[viz_data["pred_label"]],
                    true_class=self.label_names[viz_data["pred_label"]],
                    attr_class=self.label_names[viz_data["pred_label"]],
                    attr_score=sum(viz_data["attributions"]),
                    convergence_score=0.0,
                    raw_input_ids=viz_data["tokens"]  
                )
            ])

                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html.data)
                logger.info(f"Kernel SHAP HTML saved to {html_path}")
            except Exception as e:
                logger.warning(f"Failed to generate HTML visualization: {e}")

        return plt

    def compute_infidelity(self, info, noise_level=0.1, n_perturb_samples=20):
        attr = np.array(info["attributions"], copy=True)
        inputs = info["inputs_mask"]
        attributions = torch.tensor(attr, dtype=torch.float32).unsqueeze(0).to(self.device)

        def perturb_fn(x):
            noise = torch.randn_like(x) * noise_level
            return noise, x + noise

        infid = infidelity(
            forward_func=info["forward_fn"],
            perturb_func=perturb_fn,
            inputs=inputs,
            attributions=attributions,
            n_perturb_samples=n_perturb_samples
        )

        logger.info(f"Infidelity score: {infid.item():.6f}")
        return infid.item()

    def analyze(self, text, save_viz=False, viz_path=None):
        logger.info("Running Kernel SHAP...")
        info = self.get_attributions(text)
        viz_data = self.generate_visualization_data(info)
        infidelity_score = self.compute_infidelity(info)

        png_path, html_path = None, None
        if save_viz and viz_path:
            base, _ = os.path.splitext(viz_path)
            png_path = base + ".png"
            html_path = base + ".html"
            self.save_visualization(viz_data, png_path, html_path)

        return {
            "attributions": info,
            "visualization": viz_data,
            "infidelity_score": infidelity_score,
            "text": text,
            "predicted_label": info["pred_cls"],
            "visualization_paths": {
                "bar": png_path,
                "html": html_path
            }
        }
