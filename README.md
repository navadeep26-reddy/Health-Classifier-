# Health-Classifier-
The Health Classifier is a Machine Learning–based system that predicts a patient’s health condition (Normal, At Risk, or Critical) using vital signs such as heart rate, oxygen level (SpO₂), temperature, and blood pressure.
Your **Health Classifier** project is a machine learning system that predicts patient health conditions. Here's what it includes:

**Purpose**: Classifies patients as Normal, At Risk, or Critical based on vital signs

**Vital Signs Features**:
- Age, heart rate, blood oxygen (SpO₂), temperature, systolic/diastolic blood pressure, respiration rate

**Key Components**:
- **HealthClassifier class**: Trains a Random Forest model on 3,000 synthetic data points
- **Classification rules**:
  - *Critical*: SpO₂ < 92% OR heart rate > 130 OR temperature > 39.5°C OR systolic BP > 180 OR respiration > 30
  - *At Risk*: SpO₂ < 95% OR heart rate > 100 OR temperature > 38.5°C OR systolic BP > 160 OR diastolic BP > 110
  - *Normal*: Everything else
- **`predict()` method**: Returns the predicted condition, confidence score, and probability distribution
- **Interactive CLI**: `run_app()` function provides a command-line interface for real-time predictions
