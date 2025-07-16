import logging
import os
from utils.model import SentimentModel
from explainers.integrated_gradients import IntegratedGradientsAnalyzer
from explainers.layer_integrated_gradients import LayerIntegratedGradientsAnalyzer
from explainers.layer_conductance import LayerConductanceAnalyzer
from explainers.neuron_conductance import NeuronConductanceAnalyzer
from explainers.kernel_shap import KernelShapAnalyzer

logger = logging.getLogger(__name__)

def run_xai_analysis(text, target_label, neuron_idx=50):
    logger.info("Initializing model and analyzers...")
    model_handler = SentimentModel()
    model = model_handler.get_model()
    tokenizer = model_handler.get_tokenizer()
    label_names = model_handler.label_names
    device = model_handler.get_device()

    # Initialize results dictionary
    results = {
        "infidelity_scores": {},
        "visualizations": {}
    }

    try:
        # Integrated Gradients
        logger.info("Running Integrated Gradients...")
        ig = IntegratedGradientsAnalyzer(model, tokenizer, label_names, device)
        ig_result = ig.analyze(
            text=text,
            target_label=target_label,
            save_viz=True,
            viz_path="output/ig_visualization.png"
        )
        results["infidelity_scores"]["ig"] = ig_result.get("infidelity_score")
        results["visualizations"]["ig"] = "output/ig_visualization.png"
    except Exception as e:
        logger.error(f"Error in Integrated Gradients: {e}")
        results["infidelity_scores"]["ig"] = None
        results["visualizations"]["ig"] = None

    try:
        # Layer Integrated Gradients
        logger.info("Running Layer Integrated Gradients...")
        lig = LayerIntegratedGradientsAnalyzer(model, tokenizer, label_names, device)
        lig_result = lig.analyze(
            text=text,
            target_label=target_label,
            save_viz=True,
            viz_path="output/lig_visualization.png"
        )
        results["infidelity_scores"]["lig"] = lig_result.get("infidelity_score")
        results["visualizations"]["lig"] = "output/lig_visualization.png"
        results["visualizations"]["lig_html"] = "output/lig_visualization.html"
    except Exception as e:
        logger.error(f"Error in Layer Integrated Gradients: {e}")
        results["infidelity_scores"]["lig"] = None
        results["visualizations"]["lig"] = None
        results["visualizations"]["lig_html"] = None

    try:
        # Layer Conductance
        logger.info("Running Layer Conductance...")
        lc = LayerConductanceAnalyzer(model, tokenizer, label_names, device)
        lc_result = lc.analyze(
            text=text,
            target_label=target_label,
            save_viz=True,
            viz_path="output/lc_bars.png",
            comparison_path="output/lc_comparison.png"
        )
        results["infidelity_scores"]["lc"] = lc_result.get("infidelity_score")
        results["visualizations"]["lc_bars"] = "output/lc_bars.png"
        results["visualizations"]["lc_comparison"] = "output/lc_comparison.png"
        results["visualizations"]["lc_html"] = "output/lc_visualization.html"
    except Exception as e:
        logger.error(f"Error in Layer Conductance: {e}")
        results["infidelity_scores"]["lc"] = None
        results["visualizations"]["lc_bars"] = None
        results["visualizations"]["lc_comparison"] = None
        results["visualizations"]["lc_html"] = None

    try:
        # Neuron Conductance
        logger.info(f"Running Neuron Conductance for neuron {neuron_idx}...")
        nc = NeuronConductanceAnalyzer(model, tokenizer, label_names, device)
        nc_result = nc.analyze(
            text=text,
            target_label=target_label,
            neuron_idx=neuron_idx,
            save_viz=True,
            save_dir="output",
            base_name="nc"
        )
        results["infidelity_scores"]["nc"] = nc_result.get("infidelity_score")
        viz_paths = nc_result.get("visualization_paths", {})
        results["visualizations"]["nc_bar"] = viz_paths.get("bar")
        results["visualizations"]["nc_activation"] = viz_paths.get("activation_comparison")
        results["visualizations"]["nc_heatmap"] = viz_paths.get("heatmap")
        results["visualizations"]["nc_html"] = viz_paths.get("html")
    except Exception as e:
        logger.error(f"Error in Neuron Conductance: {e}")
        results["infidelity_scores"]["nc"] = None
        results["visualizations"]["nc_bar"] = None
        results["visualizations"]["nc_activation"] = None
        results["visualizations"]["nc_heatmap"] = None
        results["visualizations"]["nc_html"] = None

    try:
        logger.info("Running Kernel SHAP...")
        ks = KernelShapAnalyzer(model, tokenizer, label_names, device)
        ks_result = ks.analyze(
            text=text,
            save_viz=True,
            viz_path="output/ks_visualization"
        )
        results["infidelity_scores"]["ks"] = ks_result.get("infidelity_score")
        viz_paths = ks_result.get("visualization_paths", {})
        results["visualizations"]["ks"] = viz_paths.get("bar")
        results["visualizations"]["ks_html"] = viz_paths.get("html")
    except Exception as e:
        logger.error(f"Error in Kernel SHAP: {e}")
        results["infidelity_scores"]["ks"] = None
        results["visualizations"]["ks"] = None
        results["visualizations"]["ks_html"] = None

    return results