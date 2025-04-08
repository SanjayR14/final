import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os

# Load dataset (ensure the file is in the same directory or uploaded to root)
df = pd.read_csv("spirulina_dataset_final.csv")

# Selected features
selected_features = [
    'Initial pH', 'Illumination Intensity', 'NaNO3', 'Inoculum Level', 
    'Culture Time', 'Seawater Medium', 'NaHCO3', 'Temperature', 
    'K2HPO4', 'Salinity', 'Aeration', 'Light Time', 'Dark Time', 
    'BG11 Medium', 'Na2CO3', 'Pond System', 'Urea', 'Trace Elements', 
    'Biomass Yield'
]

# Encode protein content into classes
def categorize_protein_content(value):
    if value == 0.5:
        return 1  # Low
    elif value == 0.75:
        return 2  # Medium
    else:
        return 3  # High

df['Protein Content'] = df['Protein Content'].apply(categorize_protein_content)

# Train model
X = df[selected_features]
y = df['Protein Content']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Spirulina Protein Predictor", layout="centered")
st.title("🌿 Spirulina Protein Content Predictor")

st.write("Enter the Spirulina cultivation parameters below to predict protein content level:")

# Input form
with st.form("input_form"):
    user_input = {}
    for feature in selected_features:
        user_input[feature] = st.number_input(f"{feature}", value=0.0)
    
    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    input_data = np.array([list(user_input.values())])
    prediction = clf.predict(input_data)[0]

    protein_map = {1: "Low", 2: "Medium", 3: "High"}
    st.success(f"🧪 Predicted Protein Content Class: **{protein_map[prediction]}**")
