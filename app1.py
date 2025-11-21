import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Universal DBSCAN Clustering App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload any CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if numeric_df.shape[1] < 2:
        st.warning("CSV must contain at least 2 numeric columns for clustering.")
    else:
        st.subheader("Numeric Columns Used for Clustering")
        st.write(numeric_df.columns.tolist())

        # Scaling the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)

        # Sidebar DBSCAN params
        st.sidebar.header("DBSCAN Settings")
        eps = st.sidebar.slider("eps (ε)", 0.1, 5.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("min_samples", 1, 20, 5)

        # DBSCAN model
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)

        df['Cluster'] = labels

        st.subheader("Cluster Labels")
        st.write(df['Cluster'].value_counts())

        # PCA for 2D visualization
        if numeric_df.shape[1] >= 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            st.subheader("Cluster Visualization (PCA 2D Plot)")
            fig, ax = plt.subplots(figsize=(8,6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=50)
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.set_title("DBSCAN PCA Plot")

            st.pyplot(fig)

        st.subheader("Final Clustered Data")
        st.write(df)

        # Download clustered CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Clustered CSV",
            data=csv,
            file_name="clustered_output.csv",
            mime="text/csv"
        )
