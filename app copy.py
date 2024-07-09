import streamlit as st
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import _tree


def get_node_coordinates(tree, feature_names):
    node_coordinates = {}
    tree_ = tree.tree_
    for node_id in range(tree_.node_count):
        if tree_.children_left[node_id] != _tree.TREE_LEAF:
            node_coordinates[node_id] = {
                "left": tree_.children_left[node_id],
                "right": tree_.children_right[node_id],
                "feature": feature_names[tree_.feature[node_id]],
                "threshold": tree_.threshold[node_id],
            }
    return node_coordinates


def get_decision_path(model, sample):
    node_indicator = model.decision_path(sample)
    leave_id = model.apply(sample)
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    node_path = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]

    decision_path = []
    for node in node_path:
        if leave_id[0] == node:
            decision_path.append((node, f"Leaf Node {node}"))
        else:
            if sample[0, feature[node]] <= threshold[node]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"
            decision_path.append(
                (
                    node,
                    f"Node {node}: (X[{feature[node]}] {threshold_sign} {threshold[node]:.2f})",
                )
            )

    return decision_path


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


st.set_option("deprecation.showPyplotGlobalUse", False)

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

# Visualizations
st.write("### Data Visualizations")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
sns.boxplot(x="creditworthy", y="age", data=data, ax=axes[0, 0])
sns.boxplot(x="creditworthy", y="income", data=data, ax=axes[0, 1])
sns.boxplot(x="creditworthy", y="loan_amount", data=data, ax=axes[0, 2])
sns.boxplot(x="creditworthy", y="loan_term", data=data, ax=axes[1, 0])
sns.boxplot(x="creditworthy", y="credit_history_length", data=data, ax=axes[1, 1])
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=axes[1, 2])
st.pyplot(fig)

# Feature Engineering
st.header("Feature Engineering")
st.markdown(
    """
Identifying and creating features that can help predict the target variable.
"""
)

data["income_per_year"] = data["income"] / data["loan_term"]
st.write("### Transformed Feature: Income per Year")
st.write(data[["income", "loan_term", "income_per_year"]].head())

# Model Training
st.header("Model Training")
st.markdown(
    """
Training a decision tree on the prepared dataset.

**Gini Coefficient:** 
The Gini coefficient is a measure of impurity or disorder used by the decision tree to evaluate splits. A lower Gini coefficient indicates a purer node, meaning it contains predominantly samples of a single class. The decision tree algorithm aims to minimize the Gini coefficient to create the most homogeneous nodes possible.
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

y_pred = clf.predict(X_test)
# st.write("### Classification Report")
# st.text(classification_report(y_test, y_pred))

# Calculate the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)  # Precision for positive class
recall = recall_score(y_test, y_pred, pos_label=1)  # Recall for positive class

# Display the metrics
st.write("### Model Performance")
st.metric(label="Accuracy", value=f"{accuracy:.2f}")
st.metric(label="Precision (Positive Class)", value=f"{precision:.2f}")
st.metric(label="Recall (Positive Class)", value=f"{recall:.2f}")

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
ax.set_ylabel("Actual")

# Move the x-axis label to the top
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()

# Adjust layout properties to restrict the width
st.pyplot(fig, use_container_width=False)

# Inference
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
    "loan_amount": 15000,
    "loan_term": 24,
    "credit_history_length": 10,
    "income_per_year": 75000 / 24,  # Ensure this transformed feature is included
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

# Trace the decision path
decision_path = get_decision_path(clf, new_customer_df.values)
st.write("### Decision Path")
for node_id, step in decision_path:
    st.write(step)

# Visualize the decision tree with highlighted nodes
st.write("### Decision Tree with Decision Path")
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["Not Creditworthy", "Creditworthy"],
    filled=True,
    rounded=True,
    ax=ax,
)

# Get the coordinates of each node
node_coordinates = get_node_coordinates(clf, X.columns)

# Highlight the decision path
for node_id, step in decision_path:
    if node_id in node_coordinates:
        node = node_coordinates[node_id]
        # Custom marker for highlighting
        ax.scatter(
            node["left"],
            node["right"],
            c="red",
            s=100,
            edgecolor="k",
            label="Decision Path",
        )

st.pyplot(fig)

# Applications
st.header("Applications")
st.markdown(
    """
Demonstrating various use cases like customer churn, fraud detection, and sentiment analysis.
"""
)

# Conclusion
st.header("Conclusion")
st.markdown(
    """
In this app, we illustrated the entire process of supervised machine learning, from data collection to model evaluation.
We hope this gives you a better understanding of how to manage machine learning projects and work effectively with data scientists.
"""
)
