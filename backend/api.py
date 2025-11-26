import sys
import os
import numpy as np
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ----------------------------------------------------
# FIX: REGISTER src/ FOLDER FOR IMPORTS
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# ----------------------------------------------------
# MODEL PATHS (NEW CLEAN STRUCTURE)
# ----------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "stacking_ensemble.pkl")
STYLO_PATH = os.path.join(BASE_DIR, "models", "stylometry_classifier.pkl")
DEBERTA_PATH = os.path.join(BASE_DIR, "models", "tinybert")

# ----------------------------------------------------
# IMPORT ENSEMBLE MODEL
# ----------------------------------------------------
from src.models.ensemble import StackingEnsemble

# ----------------------------------------------------
# FASTAPI INITIALIZATION
# ----------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend anywhere
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------
logger.info("🔄 Loading ensemble model...")

ensemble = StackingEnsemble.load_model(
    MODEL_PATH,
    STYLO_PATH,
    DEBERTA_PATH
)

logger.info("✅ Ensemble model loaded successfully.")

# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------
@app.get("/")
def home():
    return {"message": "✅ AI Authorship Verification API is running!"}


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text", "")

    if not text.strip():
        return {"error": "No text provided"}

    # Run prediction
    probs = ensemble.predict_proba([text], use_weighted=True)[0]
    pred = ensemble.predict([text], use_weighted=True)[0]
    confidence = float(max(probs))
    entropy_value = float(-sum(p * np.log(p + 1e-12) for p in probs))
    word_count = len(text.split())

    result = {
        "prediction": "AI" if pred == 1 else "Human",
        "confidence": round(confidence, 4),
        "probabilities": {
            "Human": round(float(probs[0]), 4),
            "AI": round(float(probs[1]), 4)
        },
        "entropy": entropy_value,
        "word_count": word_count
    }

    logger.info(f"Prediction: {result}")
    return result


# ----------------------------------------------------
# DEV SERVER STARTER
# ----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
