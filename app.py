import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("DBSCAN Customer Segmentation App")

# Upload CSV
uploaded_file = st.file_uploader("Upload customers.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    cols = ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']
    X = df[cols]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sidebar DBSCAN parameters
    st.sidebar.header("DBSCAN Parameters")
    eps = st.sidebar.slider("eps (ε)", 0.1, 5.0, 0.5, 0.1)
    min_samples = st.sidebar.slider("min_samples", 1, 20, 5)

    # DBSCAN model
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    df['Cluster'] = labels

    st.subheader("Cluster Output")
    st.write(df[['Cluster']].value_counts())

    # PCA for plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot
    st.subheader("DBSCAN Clustering Plot (PCA 2D View)")
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, s=50)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("DBSCAN Clustering")

    st.pyplot(fig)

    st.subheader("Clustered Data")
    st.write(df)
