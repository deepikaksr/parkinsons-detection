# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
file_path = "data/parkinsons.data"
df = pd.read_csv(file_path)

# Drop the 'name' column (not relevant for prediction)
df.drop(columns=['name'], inplace=True)

# Define features (X) and target variable (y)
X = df.drop(columns=['status'])
y = df['status']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make Predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display Results
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)


def predict_parkinsons(features):
    """
    Predicts if a person has Parkinson's disease based on input voice features.
    
    Parameters:
        features (list or numpy array): A list of 22 feature values corresponding to the dataset.
        
    Returns:
        str: "Parkinson's Detected" or "Healthy"
    """
    # Ensure the input has 22 feature values
    if len(features) != X_train.shape[1]:  # Number of features in training
        return f"Error: Input should contain exactly {X_train.shape[1]} features."

    # Convert input to pandas DataFrame with column names
    features_df = pd.DataFrame([features], columns=X.columns)

    # Scale input data using the previously fitted scaler
    features_scaled = scaler.transform(features_df)  

    # Predict using the trained model
    prediction = rf_model.predict(features_scaled)[0]

    # Return the result
    return "Parkinson's Detected" if prediction == 1 else "Healthy"

# Example input for a patient with Parkinson's Disease
sample_input_parkinsons = [119.992, 157.302, 74.997, 0.00784, 0.00007, 0.00370, 0.00554, 
                           0.01109, 0.04374, 0.426, 0.0219, 0.0313, 0.0297, 0.06545, 
                           0.02211, 21.033, 0.414783, 0.815285, -4.813031, 0.266482, 
                           2.301442, 0.284654]

# Example input for a healthy person (Hypothetical values)
sample_input_healthy = [200.00, 220.00, 180.00, 0.0030, 0.00002, 0.0020, 0.0025, 
                        0.0050, 0.0150, 0.200, 0.0100, 0.0120, 0.0115, 0.0250, 
                        0.0050, 28.000, 0.3500, 0.6000, -2.000, 0.1000, 
                        1.5000, 0.1000]

# Make predictions
result_parkinsons = predict_parkinsons(sample_input_parkinsons)
result_healthy = predict_parkinsons(sample_input_healthy)

# Display results
print("\nPrediction for sample input (Parkinson's case):", result_parkinsons)
print("\nPrediction for sample input (Healthy case):", result_healthy)
print("\n")
