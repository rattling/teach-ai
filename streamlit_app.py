import streamlit as st
from scripts.decision_tree_app import decision_tree_main
from scripts.logistic_regression_app import logistic_regression_main
from scripts.clustering_app import clustering_main

# Set up the Streamlit app
st.set_page_config(
    page_title="Supervised Machine Learning in Financial Services",
    layout="wide",
)

# Sidebar for model and use case selection
st.sidebar.title("Options")
model_type = st.sidebar.selectbox(
    "Choose a model", ["Decision Tree", "Logistic Regression", "Clustering"]
)


# Run the appropriate script based on the model type
if model_type == "Decision Tree":
    # max_depth = st.sidebar.slider(
    #     "Max Depth of Decision Tree", min_value=1, max_value=10, value=3
    # )s
    decision_tree_main()
elif model_type == "Logistic Regression":
    logistic_regression_main()
elif model_type == "Clustering":
    clustering_main()
