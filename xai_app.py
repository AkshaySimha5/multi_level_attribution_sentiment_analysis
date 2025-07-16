from fastapi import FastAPI
from pydantic import BaseModel
from xai_pipeline import run_xai_analysis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class InputData(BaseModel):
    text: str
    target_label: int  # 0 = Negative, 1 = Positive


@app.post("/analyze")
async def analyze(data: InputData):
    logger.info(f"Received input: '{data.text}' | Target label: {data.target_label} ")
    result = run_xai_analysis(data.text, data.target_label)

    return {
        "message": "XAI Analysis complete.",
        "infidelity_scores": result["infidelity_scores"],
        "visualizations": result["visualizations"]
    }