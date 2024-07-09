import streamlit as st

# Set up the Streamlit app
st.set_page_config(
    page_title="Supervised Machine Learning in Financial Services", layout="wide"
)

# Sidebar for model and use case selection
st.sidebar.title("Options")
model_type = st.sidebar.selectbox(
    "Choose a model", ["Decision Tree", "Logistic Regression", "Network Analysis"]
)


# Function to run a script based on user selection
def run_script(script_name):
    with open(script_name) as script_file:
        exec(script_file.read(), globals())


# Run the appropriate script based on the model type
if model_type == "Decision Tree":
    # max_depth = st.sidebar.slider(
    #     "Max Depth of Decision Tree", min_value=1, max_value=10, value=3
    # )
    run_script("scripts/decision_tree_app.py")
elif model_type == "Logistic Regression":
    run_script("scripts/logistic_regression_app.py")
elif model_type == "Network Analysis":
    run_script("scripts/network_analysis_app.py")
