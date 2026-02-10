import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from risk_engine import RiskEngine
from explain import Explainer
from graph import build_graph, get_suspicious_clusters

# Page Config
st.set_page_config(page_title="AI Fraud Risk Simulator", layout="wide")

# Title
st.title("🛡️ AI-Driven Fraud Risk Management Simulator")
st.markdown("### Real-time Fraud Detection on Credit Card Data (PCA Features V1-V28)")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Real-time Fraud Check", "Network Analysis", "Explainability"])

# Load Data
@st.cache_data
def load_data():
    try:
        # Load a sample if file is too big (Streamlit limit)
        # But we need the whole graph.. let's load full for now, machine should handle 20k rows easily.
        # The enhanced file is likely around 20k-30k rows if we downsampled, or 280k if full. 
        # If full, 280k rows might be slow.
        df = pd.read_csv('transactions_enhanced.csv')
        # If > 50k rows, sample for dashboard performance, but keep frauds
        if len(df) > 50000:
             return df.sample(50000, random_state=42)
        return df
    except:
        return pd.DataFrame()

df = load_data()

# Initialize Modules
@st.cache_resource
def load_engine():
    return RiskEngine()

@st.cache_resource
def load_explainer():
    return Explainer()

risk_engine = load_engine()
explainer = load_explainer()

if page == "Dashboard Overview":
    st.header("📊 Transaction Overview")
    
    if df.empty:
        st.error("No data found. Please run data_generator.py (loader) first.")
    else:
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions (Sampled)", len(df))
        col2.metric("Fraud Cases", len(df[df['is_fraud'] == 1]))
        col3.metric("Fraud Rate", f"{df['is_fraud'].mean()*100:.2f}%")
        
        # Charts
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Fraud by Synthetic Location")
            fraud_by_loc = df[df['is_fraud'] == 1]['location'].value_counts().reset_index()
            fraud_by_loc.columns = ['Location', 'Count']
            fig_loc = px.bar(fraud_by_loc, x='Location', y='Count', color='Count')
            st.plotly_chart(fig_loc, use_container_width=True)
            
        with c2:
            st.subheader("Amount Distribution (Normal vs Fraud)")
            # Log scale for amount often helps visual
            fig_hist = px.histogram(df, x='amount', color='is_fraud', nbins=50, log_y=True, title="Transaction Amounts")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Recent Transactions (Raw Features V1-V28 hidden)")
        st.dataframe(df[['amount', 'location', 'device_id', 'is_fraud']].head(10))

elif page == "Real-time Fraud Check":
    st.header("🕵️ Real-time Fraud Detection")
    st.write("Since the model uses 28 PCA features (V1-V28), manual input is impractical. Pick a random transaction to test.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎲 Load Random Legitimate Transaction"):
            st.session_state['test_tx'] = df[df['is_fraud'] == 0].sample(1).iloc[0]
            
    with col2:
        if st.button("⚠️ Load Random FRAUD Transaction"):
            st.session_state['test_tx'] = df[df['is_fraud'] == 1].sample(1).iloc[0]

    if 'test_tx' in st.session_state:
        tx = st.session_state['test_tx']
        
        st.divider()
        st.write("### Transaction Details")
        st.json({
            "Amount": tx['amount'],
            "Location": tx['location'],
            "Device ID": tx['device_id'],
            "Actual Class": "Fraud" if tx['is_fraud'] == 1 else "Legitimate"
        })
        
        if st.button("Analyze Risk Score"):
            # Prepare data for API
            # Convert series to dict
            tx_dict = tx.to_dict()
            
            result = risk_engine.predict_risk(tx_dict)
            
            # Display Result
            r_col1, r_col2 = st.columns([1, 2])
            
            with r_col1:
                st.write(f"### Risk Score")
                score = result['risk_score']
                if score < 20:
                    color = "green"
                elif score < 70:
                    color = "orange"
                else:
                    color = "red"
                    
                st.markdown(f"<h1 style='text-align: center; color: {color};'>{score}</h1>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center;'>{result['risk_level']} Risk</h3>", unsafe_allow_html=True)
                
            with r_col2:
                st.write("### Analysis Details")
                st.write(f"**Model Probability:** {result['probability']:.4f}")
                
                if result['reasons']:
                    st.warning("⚠️ Risk Factors Identified:")
                    for reason in result['reasons']:
                        st.write(f"- {reason}")
                else:
                    st.success("✅ No specific risk factors identified.")

elif page == "Network Analysis":
    st.header("🕸️ Fraud Network Graph")
    st.write("Visualize users sharing devices (Potential Fraud Rings).")
    
    if st.button("Generate Graph"):
        # Use the enhanced file
        G, suspicious_devices = build_graph('transactions_enhanced.csv')
        
        if not G:
             st.error("Could not build graph.")
        else:
            clusters = get_suspicious_clusters(G)
            st.write(f"Found **{len(clusters)}** suspicious user clusters sharing devices.")
            
            if len(clusters) > 0:
                # Draw Graph
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot only the first few large clusters to avoid clutter
                # Collect nodes from top 5 clusters
                subgraph_nodes = set()
                for c in clusters[:5]:
                    subgraph_nodes.update(c)
                
                subG = G.subgraph(subgraph_nodes)
                
                pos = nx.spring_layout(subG, k=0.15, iterations=20)
                nx.draw(subG, pos, with_labels=False, node_size=100, node_color='red', alpha=0.6, ax=ax)
                st.pyplot(fig)
                
                st.write("### Detailed Clusters")
                for i, cluster in enumerate(clusters[:5]): # Show top 5
                    st.write(f"**Cluster {i+1}**: {len(cluster)} users sharing devices.")
            else:
                st.info("No suspicious clusters found (users sharing devices) in the sample.")

elif page == "Explainability":
    st.header("🧠 Explainable AI (SHAP)")
    st.write("Understand why the model made a prediction.")
    
    if 'test_tx' in st.session_state:
        tx = st.session_state['test_tx']
        st.write("Explaining the currently loaded transaction:")
        st.json({
             "Amount": tx['amount'],
             "Actual Class": "Fraud" if tx['is_fraud'] == 1 else "Legitimate"
        })
        
        if st.button("Explain This Transaction"):
             # Explainer
             shap_values, X_transformed, feature_names = explainer.explain_transaction(tx.to_dict())
             
             if shap_values is not None:
                st.subheader("Feature Contributions")
                
                # Ensure vals is 1D
                vals = np.array(shap_values)
                if vals.ndim > 1:
                    vals = vals.ravel()
                
                # Ensure equal length (in case of mismatch)
                if len(vals) != len(feature_names):
                    st.error(f"Shape mismatch: Features {len(feature_names)}, SHAP values {len(vals)}")
                else:
                    shap_df = pd.DataFrame({
                        'Feature': feature_names,
                        'SHAP Value': vals
                    })
                    
                    shap_df['AbsValue'] = shap_df['SHAP Value'].abs()
                    shap_df = shap_df.sort_values(by='AbsValue', ascending=True).tail(10) # Top 10 features
                    
                    fig = px.bar(shap_df, x='SHAP Value', y='Feature', orientation='h', title="Top 10 Feature Contributions")
                    st.plotly_chart(fig)
             else:
                 st.error("Could not generate explanation. Model/Explainer might not be loaded.")
    else:
        st.info("Please go to 'Real-time Fraud Check' tab and load a transaction first.")
