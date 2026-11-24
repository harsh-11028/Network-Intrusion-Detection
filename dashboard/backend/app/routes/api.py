"""
FastAPI routes — health, models, metrics, thresholds, insights, predict.
"""
import io
import csv
import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import MODEL_DISPLAY, RAW_INPUT_FEATURES
from app.services.model_loader import ArtifactStore, get_store
from app.services.prediction_service import predict

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# ── Health ──────────────────────────────────────────────────────────────────

@router.get("/health")
def health(store: ArtifactStore = Depends(get_store)):
    return {
        "status": "ok",
        "models_loaded": list(store.models.keys()),
        "n_raw_features": len(RAW_INPUT_FEATURES),
        "n_transformed_features": store.feature_metadata["count"],
    }


# ── Models ───────────────────────────────────────────────────────────────────

@router.get("/models")
def list_models(store: ArtifactStore = Depends(get_store)):
    result = []
    for key, disp in MODEL_DISPLAY.items():
        entry = {
            "key": key,
            **disp,
            "threshold_presets": store.threshold_presets.get(key, []),
        }
        result.append(entry)
    return {"models": result}


# ── Metrics ───────────────────────────────────────────────────────────────────

@router.get("/metrics/official")
def official_test_metrics(store: ArtifactStore = Depends(get_store)):
    """Official Phase 7.1 test metrics — never modified by uploaded CSV."""
    return {"test_metrics": store.test_metrics}


@router.get("/metrics/validation")
def validation_metrics(store: ArtifactStore = Depends(get_store)):
    return {"validation_metrics": store.validation_metrics}


@router.get("/metrics/thresholds")
def threshold_analysis(store: ArtifactStore = Depends(get_store)):
    """Full threshold sweep data for plotting."""
    return {"threshold_analysis": store.threshold_analysis.to_dict("records")}


@router.get("/metrics/test-predictions")
def test_predictions(store: ArtifactStore = Depends(get_store)):
    """y_true + all 4 model probabilities from the official test set."""
    return {"test_predictions": store.test_predictions}


# ── Insights ─────────────────────────────────────────────────────────────────

@router.get("/insights/features")
def feature_importances(store: ArtifactStore = Depends(get_store)):
    return {
        "feature_importances": store.feature_importances,
        "note": "XGBoost: gain-based from training CSV. RF/DT: sklearn feature_importances_. LR: coefficient magnitudes.",
    }


@router.get("/insights/feature-metadata")
def feature_metadata(store: ArtifactStore = Depends(get_store)):
    return {
        "raw_input_features": RAW_INPUT_FEATURES,
        "transformed_feature_names": store.feature_metadata["feature_names"],
        "n_raw": len(RAW_INPUT_FEATURES),
        "n_transformed": store.feature_metadata["count"],
    }


# ── Predict ───────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    model_key: str = Form("xgboost"),
    threshold: float = Form(0.5),
    store: ArtifactStore = Depends(get_store),
):
    """
    POST multipart/form-data:
      - file: CSV file (42 feature columns required; label/attack_cat optional)
      - model_key: logistic_regression | decision_tree | random_forest | xgboost
      - threshold: float in (0,1)
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    try:
        result = predict(store, df, model_key, threshold)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return result


@router.post("/predict/export")
async def export_predictions(
    file: UploadFile = File(...),
    model_key: str = Form("xgboost"),
    threshold: float = Form(0.5),
    store: ArtifactStore = Depends(get_store),
):
    """Same as /predict but returns a CSV file for download."""
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    try:
        result = predict(store, df, model_key, threshold)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    rows = result["rows"]
    if not rows:
        raise HTTPException(status_code=422, detail="No predictions generated.")

    output = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    filename = f"predictions_{model_key}_th{threshold:.2f}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
