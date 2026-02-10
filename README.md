# AI-Driven Fraud Risk Management Simulator (FRMS)

This is a proof-of-concept simulator demonstrating a modern fraud detection pipeline using a **Hybrid Dataset approach**.

It combines:
1.  **Real-world Credit Card Transactions** (Kaggle Dataset) for realistic fraud patterns (Features `V1`...`V28`).
2.  **Synthetic Metadata** (User IDs, Devices, Locations) to simulate Graph Network relationships and behavioral risk signals.

## Features
1.  **Hybrid Data Loader**: Downloads real data and augments it with synthetic metadata (`transactions_enhanced.csv`).
2.  **Behavioral Fraud Model**: Random Forest model to predict fraud probability based on real transaction features.
3.  **Adaptive Risk Engine**: Fuses Model Score with Rule-based checks (High Amount, Device Change, Location).
4.  **Graph Analysis**: Visualizing fraud rings (Users sharing devices).
5.  **Explainable AI**: Using SHAP to explain individual predictions.
6.  **Streamlit Dashboard**: A unified interface for the entire system.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate/Download Data
Run the data loader to fetch the real dataset and generate synthetic metadata.
```bash
python data_generator.py
```
*Note: This might take a moment to download the dataset.*

### 3. Train Model
Train the fraud detection model on the new dataset and save artifacts (`fraud_model.pkl`).
```bash
python model.py
```

### 4. Run Dashboard
Launch the Streamlit application.
```bash
python -m streamlit run app.py
```

## Dashboard Guide
- **Real-time Fraud Check**: Since the model uses 28 PCA features, use the **"Random Transaction"** buttons to load real test cases (Legitimate or Fraud) and analyze them.
- **Network Analysis**: Click "Generate Graph" to see users who share devices (potential fraud rings).
- **Explainability**: View SHAP plots to see which features (`V14`, `V4`, `Amount`...) contributed most to the fraud score.
