from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("health_model.pkl")

encoder = joblib.load("label_encoder.pkl")

features = [
    'age',
    'heart_rate',
    'spo2',
    'temperature',
    'systolic_bp',
    'diastolic_bp',
    'respiration_rate'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    values = [
        float(request.form['age']),
        float(request.form['heart_rate']),
        float(request.form['spo2']),
        float(request.form['temperature']),
        float(request.form['systolic_bp']),
        float(request.form['diastolic_bp']),
        float(request.form['respiration_rate'])
    ]

    df = pd.DataFrame([values], columns=features)

    prediction = model.predict(df)[0]

    probabilities = model.predict_proba(df)[0]

    result = encoder.inverse_transform([prediction])[0]

    confidence = round(np.max(probabilities) * 100, 2)

    return render_template(
        'index.html',
        prediction=result,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)