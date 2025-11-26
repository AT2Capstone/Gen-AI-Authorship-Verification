from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ensemble import StackingEnsemble, read_texts_from_file, entropy  # Your existing code

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(title="AI Authorship Verification API")

# Allow React frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Pydantic model for input
# -----------------------------
class TextInput(BaseModel):
    text: str

# -----------------------------
# Load Ensemble Model
# -----------------------------
ensemble_model_path = 'models/saved/stacking_ensemble.pkl'
stylometry_model_path = 'models/saved/stylometry_classifier.pkl'
deberta_model_path = 'models/saved/tinybert'

ensemble = StackingEnsemble.load_model(
    ensemble_model_path, stylometry_model_path, deberta_model_path
)

# -----------------------------
# Predict endpoint
# -----------------------------
@app.post("/predict")
def predict(input: TextInput):
    text = [input.text]  # Single sample
    probs = ensemble.predict_proba(text, use_weighted=True, weight_deberta=0.8, threshold=0.6)
    pred = ensemble.predict(text, use_weighted=True)[0]

    return {
        "prediction": "Human" if pred == 0 else "AI",
        "confidence": float(max(probs[0])),
        "entropy": float(entropy(probs[0])),
        "word_count": len(input.text.split())
    }

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"message": "Authorship Verification API is running!"}
















