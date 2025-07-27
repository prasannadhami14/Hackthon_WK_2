# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn
import tensorflow as tf

# ------------------------------
# 1. Load the trained model
# ------------------------------
MODEL_PATH = "model/blink_disease.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" Model loaded successfully.")
except Exception as e:
    print(" Error loading model:", e)
    model = None

# ------------------------------
# 2. Initialize FastAPI app
# ------------------------------
app = FastAPI(
    title="Eye Health Prediction API",
    description="API for predicting eye health issues based on blink and gaze features.",
    version="1.0.0"
)

# ------------------------------
# 3. Enable CORS
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# 4. Input schema (all numeric, including colors)
# ------------------------------
class BlinkFeatures(BaseModel):
    blink_rate: float
    avg_blink_duration: float
    gaze_stability: float
    screen_distance: float
    left_eye_color: float      # already encoded numerically
    right_eye_color: float     # already encoded numerically

# ------------------------------
# 5. Health route
# ------------------------------
@app.get("/")
def read_root():
    return {
        "message": "API is running. POST to /predict with the 6 numeric features.",
        "features_order": [
            "blink_rate",
            "avg_blink_duration",
            "gaze_stability",
            "screen_distance",
            "left_eye_color",
            "right_eye_color"
        ]
    }

# ------------------------------
# 6. Prediction route
# ------------------------------
@app.post("/predict")
def predict_eye_health(features: BlinkFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        feature_vector = [
            features.blink_rate,
            features.avg_blink_duration,
            features.gaze_stability,
            features.screen_distance,
            features.left_eye_color,
            features.right_eye_color
        ]

        X = np.array([feature_vector], dtype=np.float32)

        # Model prediction
        y_pred = model.predict(X)

        # Multiclass: shape = (1, n_classes)
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            probs = y_pred[0].astype(float).tolist()
            pred_idx = int(np.argmax(y_pred, axis=1)[0])
            confidence = float(np.max(y_pred, axis=1)[0])
            return {
                "class_index": pred_idx,
                "confidence": round(confidence, 4),
                "probabilities": probs
            }

        # Binary: shape = (1, 1)
        prob = float(y_pred[0][0])
        pred_label = int(prob >= 0.5)
        confidence = prob if pred_label == 1 else 1.0 - prob
        return {
            "label": pred_label,
            "score": round(prob, 4),
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Prediction failed", "details": str(e)}
        )

# ------------------------------
# 7. Run (dev mode)
# ------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
