try:
    import shap
except ImportError:
    shap = None

import joblib
import pandas as pd
import matplotlib.pyplot as plt

class Explainer:
    def __init__(self, model_path='fraud_model.pkl', data_path='transactions_enhanced.csv'):
        if shap is None:
            print("SHAP library not found. Explainability will be disabled.")
            self.explainer = None
            return

        try:
            self.model = joblib.load(model_path)
            # Create a background dataset for SHAP (using a sample)
            df = pd.read_csv(data_path)
            
            # Features used in model
            feature_cols = ['amount'] + [f'V{i}' for i in range(1, 29)]
            
            self.X_sample = df[feature_cols].sample(100, random_state=42)
            
            # We need the preprocessor from the pipeline to transform data
            # The model is a Pipeline: steps=['preprocessor', 'classifier']
            self.preprocessor = self.model.named_steps['preprocessor']
            self.classifier = self.model.named_steps['classifier']
            
            # Transform the sample data
            self.X_sample_transformed = self.preprocessor.transform(self.X_sample)
            
            # Initialize KernelExplainer (works for any model)
            # Using TreeExplainer for Random Forest
            # We pass the classifier and the transformed background data (optional for TreeExplainer but good for expected value)
            self.explainer = shap.TreeExplainer(self.classifier)

        except Exception as e:
            print(f"Error initializing Explainer: {e}")
            self.explainer = None

    def explain_transaction(self, transaction_data):
        if not self.explainer:
            return None, None, None
            
        # Prepare input
        # Ensure transaction_data has all features
        model_features = ['amount'] + [f'V{i}' for i in range(1, 29)]
        input_data = {k: transaction_data.get(k, 0) for k in model_features}
        
        df = pd.DataFrame([input_data])
        
        # Transform input
        X_transformed = self.preprocessor.transform(df)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_transformed)
        
        # Get feature names 
        feature_names = model_features
        
        # Determine correct SHAP values structure
        # shap_values can be a list [class0, class1] OR a single array (n_samples, n_features)
        
        target_shap = None
        
        if isinstance(shap_values, list):
            # For binary classification, usually index 1 is the positive class
            if len(shap_values) > 1:
                target_shap = shap_values[1]
            else:
                target_shap = shap_values[0]
        else:
            # It's an array. Check dimensions.
            # Shape might be (n_samples, n_features, n_classes) for some versions
            if len(shap_values.shape) == 3:
                # Assuming index 1 is fraud class
                target_shap = shap_values[:, :, 1]
            else:
                target_shap = shap_values

        # If we passed a single sample, target_shap might be (1, n_features) or just (n_features)
        # We generally want to return the values for this specific sample
        
        # If it is (1, n_features), take the first row
        if len(target_shap.shape) > 1 and target_shap.shape[0] == 1:
             target_shap_sample = target_shap[0]
        else:
             target_shap_sample = target_shap

        return target_shap_sample, X_transformed, feature_names

    def plot_shap(self, shap_values, features, feature_names):
        # This function generates a plot object or figure
        # For Streamlit, we might want to return the plot interactively
        p = shap.force_plot(self.explainer.expected_value[1], shap_values, features, feature_names=feature_names, matplotlib=True)
        return p
