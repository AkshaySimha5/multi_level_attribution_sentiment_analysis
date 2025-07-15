import logging
from utils.model import SentimentModel
from explainers.integrated_gradients import IntegratedGradientsAnalyzer
from explainers.layer_integrated_gradients import LayerIntegratedGradientsAnalyzer
from explainers.layer_conductance import LayerConductanceAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize model and tokenizer
    model_handler = SentimentModel()
    model = model_handler.get_model()
    tokenizer = model_handler.get_tokenizer()
    label_names = model_handler.label_names
    device = model_handler.get_device()

    # Input text and label
    test_text = (
        "I was skeptical about the new phone, expecting poor performance, "
        "but its fast processor and stunning display completely changed my mind."
    )
    target_label = 1  # Positive
    
    # ───────────────────────────────────────────────────────
    # Integrated Gradients Analyzer
    # ───────────────────────────────────────────────────────
    logger.info("Starting Integrated Gradients Analysis...")
    ig_analyzer = IntegratedGradientsAnalyzer(
        model=model,
        tokenizer=tokenizer,
        label_names=label_names,
        device=device
    )
    ig_result = ig_analyzer.analyze(
        text=test_text,
        target_label=target_label,
        save_viz=True,
        viz_path="output/ig_visualization.png"
    )
    logger.info(f"Integrated Gradients analysis complete. Infidelity score: {ig_result['infidelity_score']:.4f}")

    # ───────────────────────────────────────────────────────
    # Layer Integrated Gradients Analyzer
    # ───────────────────────────────────────────────────────
    logger.info("Starting Layer Integrated Gradients Analysis...")
    lig_analyzer = LayerIntegratedGradientsAnalyzer(
        model=model,
        tokenizer=tokenizer,
        label_names=label_names,
        device=device
    )
    lig_result = lig_analyzer.analyze(
        text=test_text,
        target_label=target_label,
        save_viz=True,
        viz_path="output/lig_visualization.png"
    )
    logger.info(f"Layer Integrated Gradients analysis complete. Infidelity score: {lig_result['infidelity_score']:.4f}")

    
    # ───────────────────────────────────────────────────────
    # Layer Conductance Analyzer
    # ───────────────────────────────────────────────────────
    logger.info("Starting Layer Conductance Analysis...")
    lc_analyzer = LayerConductanceAnalyzer(
        model=model,
        tokenizer=tokenizer,
        label_names=label_names,
        device=device
    )

    # Run analysis with both bar chart and comparison visualizations
    lc_result = lc_analyzer.analyze(
        text=test_text,
        target_label=target_label,
        save_viz=True,
        viz_path="output/lc_bars.png",
        comparison_path="output/lc_comparison.png"
    )

    # Access the infidelity score from the returned dictionary
    logger.info(f"Layer Conductance analysis complete. Infidelity score: {lc_result['infidelity_score']:.4f}")

    # ───────────────────────────────────────────────────────
    # Summary
    # ───────────────────────────────────────────────────────
    logger.info("="*60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*60)
    logger.info(f"Text analyzed: {test_text}")
    logger.info(f"Target label: {label_names[target_label]}")
    logger.info(f"Layer Conductance Infidelity: {lc_result['infidelity_score']:.4f}")
    logger.info(f"Integrated Gradients Infidelity: {ig_result['infidelity_score']:.4f}")
    logger.info(f"Layer Integrated Gradients Infidelity: {lig_result['infidelity_score']:.4f}")
    logger.info("All visualizations saved to output/ directory")
    logger.info("="*60)

if __name__ == "__main__":
    main()