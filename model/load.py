# load_model.py

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from eye_tracking import get_realtime_eye_metrics  # <- Import your tracker function

# Load the trained model
model = tf.keras.models.load_model("models/blink_disease.h5")

# Get real-time eye metrics
metrics = get_realtime_eye_metrics()

# Extract and reshape values
input_data = np.array([[metrics["blink_rate"],
                        metrics["avg_blink_duration"],
                        metrics["gaze_stability"],
                        metrics["screen_distance"]]])

# Apply the same preprocessing (scaling) as during training
# ⚠️ Ideally, you should have saved and loaded the same scaler, but here we recreate it for example
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_data)  # Use .transform(...) if you saved the scaler

# Predict
prediction = model.predict(input_scaled)
predicted_class = prediction.argmax(axis=1)[0]

# Map back to label if you want (optional)
label_map = {0: "dry_eye", 1: "eye_strain", 2: "normal"}
print(f"🧠 Predicted Eye Condition: {label_map[predicted_class]}")
