"""
Stage 4 — Model Serving
Minimal FastAPI inference endpoint for the trained regression pipeline.

Usage (from projects/ directory):
    python 03_mlops/serve.py

Endpoints:
    GET  /health   → service status + expected feature names
    POST /predict  → returns a regression prediction
"""
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ── Config & model ────────────────────────────────────────────────────────────

def _load_config() -> dict:
    for search in [Path.cwd(), Path.cwd().parent]:
        p = search / "config.yaml"
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("config.yaml not found. Run from the projects/ directory.")


_config   = _load_config()
_model    = joblib.load(_config["mlops"]["model_path"])
# Access feature names from the preprocessor (ColumnTransformer) directly —
# it was fitted on the named DataFrame and reliably stores input feature names.
# Pipeline.__getattr__ proxies to the last step (regressor), which was fitted
# on a numpy array and may not have feature_names_in_ set.
_features = list(_model.named_steps["preprocessor"].feature_names_in_)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Regression Pipeline API",
    description="Serves predictions from the trained regression pipeline.",
    version="1.0.0",
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: dict

    model_config = {
        "json_schema_extra": {
            "example": {"features": {"Feature_3": 1.2, "Feature_7": -0.5}}
        }
    }


class PredictResponse(BaseModel):
    prediction: float
    model_used: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "expected_features": _features}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    missing = set(_features) - set(req.features)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {sorted(missing)}",
        )

    X    = pd.DataFrame([{f: req.features[f] for f in _features}])
    pred = float(_model.predict(X)[0])

    return PredictResponse(
        prediction=pred,
        model_used=_config["mlops"]["model_path"],
    )


if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host=_config["mlops"]["host"],
        port=_config["mlops"]["port"],
        reload=True,
    )
