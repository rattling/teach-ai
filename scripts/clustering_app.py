import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


@st.cache_data
def load_data(file_path):
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)


# K-Means clustering
def kmeans_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans


# Generate textual descriptions for each cluster
def generate_cluster_descriptions(data, cluster_means):
    descriptions = []
    region_map = {1: "Rural", 2: "Suburban", 3: "Urban"}
    for cluster_id, cluster_data in cluster_means.iterrows():
        description = f"Cluster {cluster_id}:"
        description += f"\n- Average Age: {cluster_data['Age']:.1f} years"
        description += f"\n- Average Annual Income: ${cluster_data['Annual Income (k$)']:.1f}k"
        description += f"\n- Average Spending Score: {cluster_data['Spending Score (1-100)']:.1f}"
        description += f"\n- Predominant Region: {region_map[round(cluster_data['Region'])]}"
        description += f"\n- Interest in Travel: {cluster_data['Interest in Travel']:.1f}/100"
        description += f"\n- Interest in Video Games: {cluster_data['Interest in Video Games']:.1f}/100"
        description += f"\n- Interest in Sports: {cluster_data['Interest in Sports']:.1f}/100"
        description += f"\n- Interest in Dining: {cluster_data['Interest in Dining']:.1f}/100"
        description += f"\n- Interest in Gardening: {cluster_data['Interest in Gardening']:.1f}/100"
        description += f"\n- Interest in Music: {cluster_data['Interest in Music']:.1f}/100"
        description += f"\n- Average Household Income: ${cluster_data['Household Income (k$)']:.1f}k"
        descriptions.append(description)
    return descriptions


# Plot clusters
def plot_clusters(data, kmeans, num_clusters):
    pca = PCA(2)
    pca_data = pca.fit_transform(data)
    pca_data = pd.DataFrame(pca_data, columns=["PCA1", "PCA2"])
    pca_data["Cluster"] = kmeans.labels_

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="PCA1",
        y="PCA2",
        hue="Cluster",
        palette="viridis",
        data=pca_data,
        legend="full",
    )
    plt.title(f"Customer Segments with {num_clusters} Clusters")
    st.pyplot(plt.gcf())
    plt.clf()


# Create a heatmap of deviations from overall average
def plot_deviation_heatmap(data, cluster_means):
    overall_means = data.mean()
    deviations = (cluster_means - overall_means) / overall_means
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        deviations, annot=True, cmap="coolwarm", center=0, linewidths=0.5
    )
    plt.title("Deviation of Cluster Averages from Overall Average")
    st.pyplot(plt.gcf())
    plt.clf()


# Plot attribute distributions by cluster
def plot_attribute_distributions(data, features):
    for feature in features:
        plt.figure(figsize=(10, 6))
        if feature == "Region":
            sns.countplot(
                x="Cluster", hue=feature, data=data, palette="viridis"
            )
            plt.title(f"Distribution of {feature} by Cluster")
            st.pyplot(plt.gcf())
            plt.clf()
        else:
            sns.boxplot(x="Cluster", y=feature, data=data, palette="viridis")
            plt.title(f"Distribution of {feature} by Cluster")
            st.pyplot(plt.gcf())
            plt.clf()


# Streamlit app
def clustering_main():
    st.title("Customer Segmentation Using K-Means Clustering")
    st.write(
        """
    This app demonstrates customer segmentation using K-Means clustering.
    You can adjust the number of clusters using the slider below and see the resulting segments.
    """
    )

    # Load data
    data = load_data("data/clustering.csv")

    if not data.empty:
        # Select number of clusters
        num_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 5)

        # Select features for clustering
        features = [
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)",
            "Region",
            "Interest in Travel",
            "Interest in Video Games",
            "Interest in Sports",
            "Interest in Dining",
            "Interest in Gardening",
            "Interest in Music",
            "Household Income (k$)",
        ]

        # Perform clustering
        clusters, kmeans = kmeans_clustering(data[features], num_clusters)
        data["Cluster"] = clusters

        # Display data
        st.subheader("Clustered Data")
        st.write(data)

        # Legend for Region
        st.markdown(
            """
        **Region Legend:**
        - **1**: Rural
        - **2**: Suburban
        - **3**: Urban
        """
        )

        # Plot clusters
        plot_clusters(data[features], kmeans, num_clusters)

        # Cluster descriptions
        st.subheader("Cluster Descriptions")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        cluster_means = data.groupby("Cluster")[numeric_cols].mean()

        # Plot heatmap of deviations
        plot_deviation_heatmap(data[features], cluster_means)

        # Generate and display textual descriptions of clusters
        descriptions = generate_cluster_descriptions(data, cluster_means)
        st.subheader("Cluster Interpretations")
        for desc in descriptions:
            st.text(desc)

        # Plot attribute distributions by cluster
        st.subheader("Attribute Distributions by Cluster")
        plot_attribute_distributions(data, features)
