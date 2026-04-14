from flask import Flask, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load model and bands
model = joblib.load("model.pkl")
bands = joblib.load("bands.pkl")

# Load image
X_img = joblib.load("X_img.pkl")   # shape (H, W, B)

@app.route('/')
def home():
    return "Model is running ✅"

@app.route('/predict', methods=['GET'])
def predict():
    H, W, B = X_img.shape

    # Reshape
    X = X_img.reshape(-1, B)

    # Select important bands
    X = X[:, bands]

    # Predict
    y_pred = model.predict(X)

    # Reshape back
    pred_map = y_pred.reshape(H, W)

    return jsonify(pred_map.tolist())

if __name__ == '__main__':
    app.run(debug=True)