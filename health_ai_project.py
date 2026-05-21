import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

n = 5000

age = np.random.randint(18, 90, n)

heart_rate = np.random.normal(75, 15, n).clip(50, 160)

spo2 = np.random.normal(97, 2, n).clip(85, 100)

temperature = np.random.normal(36.8, 0.5, n).clip(35, 41)

systolic = np.random.normal(120 + age*0.1, 15, n).clip(90, 200)

diastolic = np.random.normal(80 + age*0.05, 10, n).clip(50, 130)

respiration = np.random.normal(16, 3, n).clip(10, 35)

conditions = []

for i in range(n):

    if (
        spo2[i] < 92 or
        heart_rate[i] > 130 or
        temperature[i] > 39.5 or
        systolic[i] > 180 or
        respiration[i] > 30
    ):
        conditions.append("Critical")

    elif (
        spo2[i] < 95 or
        heart_rate[i] > 100 or
        temperature[i] > 38.5 or
        systolic[i] > 160 or
        diastolic[i] > 110
    ):
        conditions.append("At Risk")

    else:
        conditions.append("Normal")

df = pd.DataFrame({
    'age': age,
    'heart_rate': heart_rate,
    'spo2': spo2,
    'temperature': temperature,
    'systolic_bp': systolic,
    'diastolic_bp': diastolic,
    'respiration_rate': respiration,
    'condition': conditions
})

features = [
    'age',
    'heart_rate',
    'spo2',
    'temperature',
    'systolic_bp',
    'diastolic_bp',
    'respiration_rate'
]

X = df[features]

encoder = LabelEncoder()

y = encoder.fit_transform(df['condition'])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10
)

model.fit(X_train, y_train)

joblib.dump(model, "health_model.pkl")

joblib.dump(encoder, "label_encoder.pkl")

print("Model Saved Successfully!")