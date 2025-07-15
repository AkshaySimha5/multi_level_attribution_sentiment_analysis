# xai_pipeline.py

import logging
from utils.model import SentimentModel
from explainers.integrated_gradients import IntegratedGradientsAnalyzer
from explainers.layer_integrated_gradients import LayerIntegratedGradientsAnalyzer
from explainers.layer_conductance import LayerConductanceAnalyzer

logger = logging.getLogger(__name__)

def run_xai_analysis(text, target_label):
    logger.info("Initializing model and analyzers...")
    model_handler = SentimentModel()
    model = model_handler.get_model()
    tokenizer = model_handler.get_tokenizer()
    label_names = model_handler.label_names
    device = model_handler.get_device()

    logger.info("Running Integrated Gradients...")
    ig = IntegratedGradientsAnalyzer(model, tokenizer, label_names, device)
    ig_result = ig.analyze(
        text=text,
        target_label=target_label,
        save_viz=True,
        viz_path="output/ig_visualization.png"
    )

    logger.info("Running Layer Integrated Gradients...")
    lig = LayerIntegratedGradientsAnalyzer(model, tokenizer, label_names, device)
    lig_result = lig.analyze(
        text=text,
        target_label=target_label,
        save_viz=True,
        viz_path="output/lig_visualization.png"
    )

    logger.info("Running Layer Conductance...")
    lc = LayerConductanceAnalyzer(model, tokenizer, label_names, device)
    lc_result = lc.analyze(
        text=text,
        target_label=target_label,
        save_viz=True,
        viz_path="output/lc_bars.png",
        comparison_path="output/lc_comparison.png"
    )

    return {
        "infidelity_scores": {
            "ig": ig_result["infidelity_score"],
            "lig": lig_result["infidelity_score"],
            "lc": lc_result["infidelity_score"]
        },
        "visualizations": {
            "ig": "output/ig_visualization.png",
            "lig": "output/lig_visualization.png",
            "lc_bars": "output/lc_bars.png",
            "lc_comparison": "output/lc_comparison.png",
            "lig_html": "output/lig_visualization.html",
            "lc_html": "output/lc_visualization.html"
        }
    }
