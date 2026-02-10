import joblib
import pandas as pd
import numpy as np

class RiskEngine:
    def __init__(self, model_path='fraud_model.pkl'):
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Error: Model file {model_path} not found.")
            self.model = None

    def predict_risk(self, transaction_data):
        """
        Predicts fraud probability and calculates a risk score.
        transaction_data: dict containing 'amount', 'V1'...'V28', 'location', 'device_id'
        """
        if not self.model:
            return {'error': 'Model not loaded'}

        # Prepare input DataFrame for Model (Only real features)
        # Model expects: amount, V1..V28
        model_features = ['amount'] + [f'V{i}' for i in range(1, 29)]
        
        # Ensure all model features are present
        input_data = {k: transaction_data.get(k, 0) for k in model_features}
        df = pd.DataFrame([input_data])
        
        # Get Model Probability
        if hasattr(self.model, 'predict_proba'):
            prob_fraud = self.model.predict_proba(df)[0][1] # Probability of class 1 (Fraud)
        else:
            prob_fraud = float(self.model.predict(df)[0])
        
        # Base Risk Score (0-100) from Model Probability
        risk_score = prob_fraud * 100
        
        # --- Rule-Based Adjustments (Hybrid Approach) ---
        # Using Real 'Amount' and Synthetic 'Location'/'Device'
        
        reasons = []
        
        # Rule 1: High Amount Check (Real feature)
        # Real dataset amounts are often small, but let's say > 3000 is high
        if transaction_data.get('amount', 0) > 3000:
            risk_score = max(risk_score, 80) 
            reasons.append("High Transaction Amount (> $3000)")
        
        # Rule 2: Suspicious Location (Synthetic feature)
        # We simulate this check
        if transaction_data.get('location') not in ['New York', 'London', 'Berlin', 'Tokyo'] and risk_score > 20: 
             risk_score += 15
             reasons.append("Uncommon Location for User Profile")
             
        # Rule 3: Device Change (Synthetic)
        # In a real app, we'd check history. Here we just simulate a flag if provided
        if transaction_data.get('is_new_device', False):
            risk_score += 20
            reasons.append("New Device Detected")
        
        # Cap Score at 100
        risk_score = min(risk_score, 100)
        
        # Determine Risk Level
        if risk_score < 20:
            risk_level = "Low"
        elif risk_score < 70:
            risk_level = "Medium"
        else:
            risk_level = "High"
            
        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'probability': round(prob_fraud, 4),
            'reasons': reasons
        }

if __name__ == "__main__":
    engine = RiskEngine()
    # Dummy V features
    sample_tx = {f'V{i}': 0.1 for i in range(1, 29)}
    sample_tx['amount'] = 5000
    sample_tx['location'] = 'Unknown'
    print(engine.predict_risk(sample_tx))
