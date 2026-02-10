import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def build_graph(data_path='transactions.csv'):
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        return None

    G = nx.Graph()
    
    # We want to link Users who share the same Device
    # Nodes: Users
    # Edges: Shared Device
    
    # Group by device_id
    device_groups = df.groupby('device_id')['user_id'].unique()
    
    suspicious_devices = []
    
    for device, users in device_groups.items():
        if len(users) > 1:
            suspicious_devices.append(device)
            # Create edges between all users sharing this device
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    G.add_edge(users[i], users[j], device=device)
    
    return G, suspicious_devices

def get_suspicious_clusters(G):
    if not G:
        return []
    
    clusters = list(nx.connected_components(G))
    # Filter for larger clusters which might indicate a fraud ring
    fraud_rings = [cluster for cluster in clusters if len(cluster) > 1]
    return fraud_rings
