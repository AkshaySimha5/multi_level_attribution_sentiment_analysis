# run_demo.py

import logging
from xai_pipeline import run_xai_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    test_text = (
        "I was skeptical about the new phone, expecting poor performance, but its fast processor and stunning display completely changed my mind."
    )
    target_label = 1  # 0 = Negative, 1 = Positive

    result = run_xai_analysis(test_text, target_label)

    logger.info("="*60)
    logger.info("XAI ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Text: {test_text}")
    logger.info(f"Target label: {target_label}")
    logger.info(f"Infidelity Scores: {result['infidelity_scores']}")
    logger.info(f"Visualizations saved at:")
    for key, path in result["visualizations"].items():
        logger.info(f"  {key}: {path}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
