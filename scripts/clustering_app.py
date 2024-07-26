import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# Function to create the dataset
def load_data():
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "CustomerID": range(1, 301),
            "Age": np.concatenate(
                [
                    np.random.randint(60, 80, 50),  # Luxury Retirees
                    np.random.randint(30, 50, 50),  # Active Gamers
                    np.random.randint(18, 25, 50),  # Urban Sport Enthusiasts
                    np.random.randint(30, 40, 50),  # Gourmet Gardeners
                    np.random.randint(25, 45, 100),  # Tech-Savvy Remote Workers
                ]
            ),
            "Annual Income (k$)": np.concatenate(
                [
                    np.random.randint(70, 150, 50),  # Luxury Retirees
                    np.random.randint(40, 100, 50),  # Active Gamers
                    np.random.randint(10, 30, 50),  # Urban Sport Enthusiasts
                    np.random.randint(40, 80, 50),  # Gourmet Gardeners
                    np.random.randint(50, 120, 100),  # Tech-Savvy Remote Workers
                ]
            ),
            "Spending Score (1-100)": np.concatenate(
                [
                    np.random.randint(40, 80, 50),  # Luxury Retirees
                    np.random.randint(60, 90, 50),  # Active Gamers
                    np.random.randint(20, 60, 50),  # Urban Sport Enthusiasts
                    np.random.randint(30, 70, 50),  # Gourmet Gardeners
                    np.random.randint(50, 90, 100),  # Tech-Savvy Remote Workers
                ]
            ),
            "Region": np.concatenate(
                [
                    [3] * 10 + [2] * 20 + [1] * 20,  # Luxury Retirees
                    [3] * 20 + [2] * 20 + [1] * 10,  # Active Gamers
                    [3] * 50,  # Urban Sport Enthusiasts
                    [3] * 10 + [2] * 30 + [1] * 10,  # Gourmet Gardeners
                    [3] * 50 + [2] * 50,  # Tech-Savvy Remote Workers
                ]
            ),
            "Employment Status": np.concatenate(
                [
                    ["Retired"] * 50,  # Luxury Retirees
                    ["Employed"] * 50,  # Active Gamers
                    ["Student"] * 50,  # Urban Sport Enthusiasts
                    ["Employed"] * 50,  # Gourmet Gardeners
                    ["Employed"] * 100,  # Tech-Savvy Remote Workers
                ]
            ),
            "Interest in Travel": np.concatenate(
                [
                    np.random.randint(60, 100, 50),  # Luxury Retirees
                    np.random.randint(1, 40, 50),  # Active Gamers
                    np.random.randint(1, 20, 50),  # Urban Sport Enthusiasts
                    np.random.randint(1, 30, 50),  # Gourmet Gardeners
                    np.random.randint(30, 60, 100),  # Tech-Savvy Remote Workers
                ]
            ),
            "Interest in Video Games": np.concatenate(
                [
                    np.random.randint(1, 20, 50),  # Luxury Retirees
                    np.random.randint(60, 100, 50),  # Active Gamers
                    np.random.randint(10, 40, 50),  # Urban Sport Enthusiasts
                    np.random.randint(10, 30, 50),  # Gourmet Gardeners
                    np.random.randint(50, 90, 100),  # Tech-Savvy Remote Workers
                ]
            ),
            "Interest in Sports": np.concatenate(
                [
                    np.random.randint(20, 60, 50),  # Luxury Retirees
                    np.random.randint(40, 80, 50),  # Active Gamers
                    np.random.randint(60, 100, 50),  # Urban Sport Enthusiasts
                    np.random.randint(30, 70, 50),  # Gourmet Gardeners
                    np.random.randint(30, 70, 100),  # Tech-Savvy Remote Workers
                ]
            ),
            "Interest in Dining": np.concatenate(
                [
                    np.random.randint(10, 50, 50),  # Luxury Retirees
                    np.random.randint(20, 60, 50),  # Active Gamers
                    np.random.randint(20, 50, 50),  # Urban Sport Enthusiasts
                    np.random.randint(60, 100, 50),  # Gourmet Gardeners
                    np.random.randint(40, 70, 100),  # Tech-Savvy Remote Workers
                ]
            ),
            "Interest in Gardening": np.concatenate(
                [
                    np.random.randint(20, 60, 50),  # Luxury Retirees
                    np.random.randint(10, 50, 50),  # Active Gamers
                    np.random.randint(10, 40, 50),  # Urban Sport Enthusiasts
                    np.random.randint(60, 100, 50),  # Gourmet Gardeners
                    np.random.randint(20, 50, 100),  # Tech-Savvy Remote Workers
                ]
            ),
            "Interest in Music": np.concatenate(
                [
                    np.random.randint(30, 70, 50),  # Luxury Retirees
                    np.random.randint(20, 60, 50),  # Active Gamers
                    np.random.randint(20, 50, 50),  # Urban Sport Enthusiasts
                    np.random.randint(30, 70, 50),  # Gourmet Gardeners
                    np.random.randint(50, 90, 100),  # Tech-Savvy Remote Workers
                ]
            ),
            "Household Income (k$)": np.concatenate(
                [
                    np.random.randint(70, 150, 50),  # Luxury Retirees
                    np.random.randint(40, 100, 50),  # Active Gamers
                    np.random.randint(10, 30, 50),  # Urban Sport Enthusiasts
                    np.random.randint(40, 80, 50),  # Gourmet Gardeners
                    np.random.randint(50, 120, 100),  # Tech-Savvy Remote Workers
                ]
            ),
        }
    )
    return data


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
        description += (
            f"\n- Average Annual Income: ${cluster_data['Annual Income (k$)']:.1f}k"
        )
        description += (
            f"\n- Average Spending Score: {cluster_data['Spending Score (1-100)']:.1f}"
        )
        description += (
            f"\n- Predominant Region: {region_map[round(cluster_data['Region'])]}"
        )
        description += (
            f"\n- Interest in Travel: {cluster_data['Interest in Travel']:.1f}/100"
        )
        description += f"\n- Interest in Video Games: {cluster_data['Interest in Video Games']:.1f}/100"
        description += (
            f"\n- Interest in Sports: {cluster_data['Interest in Sports']:.1f}/100"
        )
        description += (
            f"\n- Interest in Dining: {cluster_data['Interest in Dining']:.1f}/100"
        )
        description += f"\n- Interest in Gardening: {cluster_data['Interest in Gardening']:.1f}/100"
        description += (
            f"\n- Interest in Music: {cluster_data['Interest in Music']:.1f}/100"
        )
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
    sns.heatmap(deviations, annot=True, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Deviation of Cluster Averages from Overall Average")
    st.pyplot(plt.gcf())
    plt.clf()


# Plot attribute distributions by cluster
def plot_attribute_distributions(data, features):
    for feature in features:
        plt.figure(figsize=(10, 6))
        if feature == "Region":
            sns.countplot(x="Cluster", hue=feature, data=data, palette="viridis")
            plt.title(f"Distribution of {feature} by Cluster")
            st.pyplot(plt.gcf())
            plt.clf()
        else:
            sns.boxplot(x="Cluster", y=feature, data=data, palette="viridis")
            plt.title(f"Distribution of {feature} by Cluster")
            st.pyplot(plt.gcf())
            plt.clf()


# Streamlit app
def main():
    st.title("Customer Segmentation Using K-Means Clustering")
    st.write(
        """
    This app demonstrates customer segmentation using K-Means clustering.
    You can adjust the number of clusters using the slider below and see the resulting segments.
    """
    )

    # Load data
    data = load_data()

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


if __name__ == "__main__":
    main()
