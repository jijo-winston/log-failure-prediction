from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from src.config import PATHS

app = FastAPI(
    title="Log Failure Prediction API",
    version="1.0.0",
    description="FastAPI service for serving failure-risk alerts and model metadata.",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """
    Returns ensemble config + last known evaluation metrics (if present).
    """
    params_path = PATHS.models_dir / "forward_ensemble_params.json"
    train_metrics_path = PATHS.models_dir / "forward_ensemble_train_metrics.json"
    test_metrics_path = PATHS.reports_dir / "forward_ensemble_test_metrics.json"
    ew_metrics_path = PATHS.reports_dir / "forward_early_warning_metrics.json"

    return {
        "ensemble_params": _read_json(params_path) if params_path.exists() else None,
        "train_metrics": _read_json(train_metrics_path) if train_metrics_path.exists() else None,
        "test_metrics": _read_json(test_metrics_path) if test_metrics_path.exists() else None,
        "early_warning_metrics": _read_json(ew_metrics_path) if ew_metrics_path.exists() else None,
    }


@app.get("/alerts/topk")
def alerts_topk(
    k: int = Query(default=50, ge=1, le=500),
):
    """
    Returns top-k risky blocks from the latest generated alerts JSON.

    Prerequisite:
        python -m src.modeling.generate_alerts
    """
    alerts_path = PATHS.reports_dir / "alerts_topk.json"
    if not alerts_path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "error": "alerts_topk.json not found",
                "hint": "Run: python -m src.modeling.generate_alerts",
                "expected_path": str(alerts_path),
            },
        )

    payload = _read_json(alerts_path)

    # If file contains more than needed, slice safely
    alerts = payload.get("alerts", [])
    payload["alerts"] = alerts[:k]
    payload["top_k"] = k
    return payload