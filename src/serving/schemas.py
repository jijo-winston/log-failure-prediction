from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., description="Raw log text (block-level aggregated text or any log chunk).")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Decision threshold for anomaly label.")


class PredictResponse(BaseModel):
    prob_anomaly: float
    pred_label: int
    threshold: float