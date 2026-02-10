import pandas as pd
import numpy as np
import random
import uuid
import requests
import io
import os

# Configuration
DATA_URL = "https://media.githubusercontent.com/media/sharmaroshan/Credit-Card-Fraud-Detection/master/creditcard.csv"
# Backup URL in case the first one fails or is LFS pointer
BACKUP_URL = "https://raw.githubusercontent.com/amankharwal/Website-data/master/creditcard.csv"

OUTPUT_FILE = 'transactions_enhanced.csv'
NUM_USERS = 2000

def download_data():
    print("Attempting to download real credit card fraud dataset...")
    
    df = None
    for url in [DATA_URL, BACKUP_URL]:
        try:
            print(f"Downloading from {url}...")
            response = requests.get(url)
            if response.status_code == 200:
                # Check if it's an LFS pointer (small file starting with 'version')
                if response.content.startswith(b'version https://git-lfs.github.com'):
                    print("URL returned Git LFS pointer, skipping...")
                    continue
                    
                df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
                print("Download successful!")
                break
            else:
                print(f"Failed with status {response.status_code}")
        except Exception as e:
            print(f"Error downloading: {e}")

    if df is None:
        print("Could not download real data. Generating fallback synthetic data...")
        # Fallback to a small synthetic V1-V28 dataset structure
        return generate_synthetic_real_structure()
    
    return df

def generate_synthetic_real_structure():
    # Only if download fails
    count = 5000
    data = {f'V{i}': np.random.randn(count) for i in range(1, 29)}
    data['Time'] = np.random.randint(0, 172792, count)
    data['Amount'] = np.random.uniform(0, 3000, count)
    data['Class'] = np.random.choice([0, 1], count, p=[0.98, 0.02])
    return pd.DataFrame(data)

def augment_data(df):
    print("Augmenting data with synthetic metadata (User IDs, Devices, Locations)...")
    
    # Take a sample if too large to keep it fast
    if len(df) > 20000:
        # Keep all frauds, sample normal
        frauds = df[df['Class'] == 1]
        normal = df[df['Class'] == 0].sample(20000)
        df = pd.concat([frauds, normal]).sample(frac=1).reset_index(drop=True)
        print(f"Downsampled to {len(df)} rows for performance.")

    # Synthetic User Base
    users = [str(uuid.uuid4())[:8] for _ in range(NUM_USERS)]
    devices = [str(uuid.uuid4())[:10] for _ in range(NUM_USERS)]
    locations = ['New York', 'London', 'Singapore', 'Tokyo', 'San Francisco', 'Berlin', 'Sydney', 'Mumbai']
    
    # Assign Users to rows
    # We want consistent mapping for valid users, but we don't have user_id in real data.
    # Just assign randomly.
    row_users = np.random.choice(users, len(df))
    df['user_id'] = row_users
    
    # Assign Devices
    # Lookup device for user
    user_device_map = dict(zip(users, devices))
    df['device_id'] = df['user_id'].map(user_device_map)
    
    # Assign Locations
    df['location'] = np.random.choice(locations, len(df))
    
    # --- Inject Fraud Rings (Synthetic Relationships) ---
    # Find fraud rows
    fraud_indices = df[df['Class'] == 1].index
    
    # Group some frauds to share the SAME DEVICE (Fraud Ring)
    if len(fraud_indices) > 50:
        ring_size = 20
        ring_indices = fraud_indices[:ring_size]
        shared_device = "BAD_DEVICE_" + str(uuid.uuid4())[:5]
        df.loc[ring_indices, 'device_id'] = shared_device
        print(f"Injected Fraud Ring: {ring_size} fraud cases on device {shared_device}")

    # --- Feature Compatibility renaming ---
    # The real dataset uses 'Class'. Our old code used 'is_fraud'. Renaming for consistency? 
    # Or update my code to use 'Class'. 
    # Let's rename 'Class' to 'is_fraud' to minimize downstream breakage, 
    # but I still need to update model.py for V features.
    df = df.rename(columns={'Class': 'is_fraud', 'Amount': 'amount'})
    
    # Ensure Time is present (some datasets drop it)
    if 'Time' not in df.columns:
        df['Time'] = range(len(df))

    return df

def main():
    df = download_data()
    df = augment_data(df)
    
    output_path = f'c:/Users/AnujYadav/OneDrive - Arrk Group/Desktop/FRMS/{OUTPUT_FILE}'
    df.to_csv(output_path, index=False)
    print(f"Enhanced data saved to {output_path}")
    print(df.head())
    print("\nColumns:", df.columns.tolist())

if __name__ == "__main__":
    main()
