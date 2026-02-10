import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_model():
    print("Loading data...")
    try:
        df = pd.read_csv('transactions_enhanced.csv')
    except FileNotFoundError:
        print("Error: transactions_enhanced.csv not found. Please run data_generator.py first.")
        return
    
    # Define features and target
    # V1-V28 are PCA features, Amount is raw. Time is seconds (omitting for simplicity or robustness)
    feature_cols = ['amount'] + [f'V{i}' for i in range(1, 29)]
    target = 'is_fraud'
    
    # Check if columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        return

    X = df[feature_cols]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Preprocessing Pipeline
    # Scale Amount and V features (V are likely already scaled around 0, but scaling doesn't hurt)
    
    numeric_features = feature_cols # All are numeric
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ])
    
    # Model Pipeline
    # Class weight balanced for imbalanced data
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1))])
    
    print("Training model (Random Forest)... this might take a minute...")
    clf.fit(X_train, y_train)
    
    # Evaluation
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save artifacts
    print("Saving model artifacts...")
    joblib.dump(clf, 'fraud_model.pkl')
    print("Model saved to fraud_model.pkl")

if __name__ == "__main__":
    train_model()
