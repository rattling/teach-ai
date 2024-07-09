import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.patches as patches
from sklearn.tree import _tree
import graphviz
import pydotplus
from sklearn.tree import export_graphviz
from PIL import Image
from io import BytesIO

# Set up the Streamlit app
st.set_page_config(
    page_title="Supervised Machine Learning in Financial Services", layout="wide"
)

# Sidebar for dataset selection
st.sidebar.title("Options")
dataset_name = st.sidebar.selectbox("Choose a dataset", ["Credit Scoring"])

# Add max depth slider
max_depth = st.sidebar.slider(
    "Max Depth of Decision Tree", min_value=1, max_value=10, value=3
)

# Load dataset based on selection
if dataset_name == "Credit Scoring":
    data = pd.read_csv("data/credit_scoring.csv")

# Introduction
st.title("Supervised Machine Learning in Financial Services")
st.header("Introduction")
st.markdown(
    """
Welcome to this interactive app designed to illustrate the concepts of supervised machine learning for a business audience in financial services. 
Here, you'll learn the importance of labelled data, how to identify potential predictors, the process of data transformation, model training, and evaluation.
By the end of this session, you'll be better equipped to manage machine learning projects and collaborate effectively with data scientists.
"""
)

# Data Exploration
st.header("Data Exploration")
st.write("### Dataset Overview")
st.write(data.head())
st.write("### Summary Statistics")
st.write(data.describe())


# Feature Engineering
st.header("Feature Engineering")
st.markdown(
    """
Identifying and creating features that can help predict the target variable.
"""
)

# Dummify the categorical variable
data = pd.get_dummies(data, columns=["education_level"])

# Show the transformed data
st.write("### Transformed Dataset with Dummified Variables")
st.write(data.head())

# Visualizations
st.write("### Data Visualizations")
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

sns.boxplot(x="creditworthy", y="age", data=data, ax=axes[0, 0])
sns.boxplot(x="creditworthy", y="income", data=data, ax=axes[0, 1])
sns.boxplot(x="creditworthy", y="months_employed", data=data, ax=axes[0, 2])
sns.boxplot(x="creditworthy", y="avg_credit_card_bal", data=data, ax=axes[1, 0])
sns.boxplot(x="creditworthy", y="months_with_bank", data=data, ax=axes[1, 1])

# Bar plot for the dummified education_level variables
education_levels = [
    "education_level_Junior Certificate",
    "education_level_Leaving Certificate",
    "education_level_NFQ7",
    "education_level_NFQ8",
    "education_level_NFQ9",
]
data_melted = pd.melt(data, id_vars=["creditworthy"], value_vars=education_levels)
sns.barplot(
    x="variable", y="value", hue="creditworthy", data=data_melted, ax=axes[1, 2]
)

st.pyplot(fig)

# Create a new figure for the heatmap
st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)


# Model Training
st.header("Model Training")
st.markdown(
    """
Training a decision tree on the prepared dataset.
"""
)

X = data.drop("creditworthy", axis=1)
y = data["creditworthy"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train)

st.write("### Decision Tree")
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["Not Creditworthy", "Creditworthy"],
    filled=True,
    rounded=True,
    ax=ax,
)
st.pyplot(fig)

# Model Evaluation
st.header("Model Evaluation")
st.markdown(
    """
Evaluating the model on a test set.
"""
)

# Predict and evaluate
y_pred = clf.predict(X_test)

# Calculate the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)  # Precision for positive class
recall = recall_score(y_test, y_pred, pos_label=1)  # Recall for positive class

# Display the metrics
st.write("### Model Performance")
st.metric(label="Accuracy", value=f"{accuracy:.2f}")
st.metric(label="Precision (Positive Class)", value=f"{precision:.2f}")
st.metric(label="Recall (Positive Class)", value=f"{recall:.2f}")

# Detailed classification report (optional)
st.write("### Detailed Classification Report")
st.text(classification_report(y_test, y_pred))

st.write("### Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(3, 3))  # Adjust the size here
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="coolwarm",
    xticklabels=["Creditworthy", "Not Creditworthy"],
    yticklabels=["Creditworthy", "Not Creditworthy"],
    ax=ax,
    cbar=False,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

# Move the x-axis label to the top
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()

st.pyplot(fig)

# Inference
st.header("Inference on New Data")
st.markdown(
    """
Let's introduce a new customer, show their data, and make an inference with our trained model.
"""
)

# New customer data
new_customer = {
    "age": 35,
    "income": 75000,
    "months_employed": 60,
    "avg_credit_card_bal": 3000,
    "months_with_bank": 100,
    "education_level_Leaving Certificate": 1,
    "education_level_NFQ7": 0,
    "education_level_NFQ8": 0,
    "education_level_NFQ9": 0,
    "education_level_Junior Certificate": 0,
}

new_customer_df = pd.DataFrame(
    [new_customer], columns=X.columns
)  # Ensure feature names match


st.write("### New Customer Data")
st.write(new_customer_df)

# Make a prediction
prediction = clf.predict(new_customer_df)[0]
prediction_proba = clf.predict_proba(new_customer_df)[0]

# Display the prediction
prediction_label = "Creditworthy" if prediction == 1 else "Not Creditworthy"
st.write(f"### Prediction: {prediction_label}")
st.write(f"### Confidence: {prediction_proba[prediction]:.2f}")

# Visualize the confidence
st.write("### Prediction Confidence")
fig, ax = plt.subplots(figsize=(5, 3))
sns.barplot(
    x=["Not Creditworthy", "Creditworthy"],
    y=prediction_proba,
    palette="coolwarm",
    ax=ax,
)
ax.set_ylabel("Probability")
st.pyplot(fig)


# Function to get the decision path
def get_decision_path(model, sample):
    node_indicator = model.decision_path(sample)
    leaf_id = model.apply(sample)
    node_path = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]
    return node_path


# Highlight the decision path using Graphviz
st.write("### Decision Tree with Decision Path")

# Get the decision path
decision_path = get_decision_path(clf, new_customer_df.values)

# Export the decision tree to DOT format
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=X.columns,
    class_names=["Not Creditworthy", "Creditworthy"],
    filled=True,
    rounded=True,
    special_characters=True,
)

# Convert to graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Highlight the decision path
for node_id in decision_path:
    node = graph.get_node(str(node_id))[0]
    node.set_fillcolor("yellow")

# Convert graph to PNG image
image_data = graph.create_png()
image = Image.open(BytesIO(image_data))

# Display the image in Streamlit
st.image(image)
