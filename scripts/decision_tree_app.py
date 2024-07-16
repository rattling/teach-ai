import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pydotplus
from PIL import Image
from io import BytesIO
from sklearn.tree import export_graphviz

"""NOTE - Only going to support credit scoring use case for now in decision tree app. """


# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)


# Decision Tree for Credit Scoring
def credit_scoring():
    st.header("Credit Scoring with Decision Tree")
    return load_data("data/credit_scoring.csv")
    # Add feature engineering, model training, evaluation, and visualization code here


# Decision Tree for Fraud Detection
def fraud_detection():
    st.header("Fraud Detection with Decision Tree")
    return load_data("data/fraud_detection.csv")
    # Add feature engineering, model training, evaluation, and visualization code here


# Add max depth slider
max_depth = st.sidebar.slider(
    "Max Depth of Decision Tree", min_value=1, max_value=10, value=3
)

# Sidebar for use case selection
use_case = st.sidebar.selectbox("Choose a use case", ["Credit Scoring"])

# Run the appropriate function based on the use case
if use_case == "Credit Scoring":
    data = credit_scoring()
elif use_case == "Fraud Detection":
    data = fraud_detection()

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

# Create box plots
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

sns.boxplot(x="creditworthy", y="age", data=data, ax=axes[0, 0])
sns.boxplot(x="creditworthy", y="income", data=data, ax=axes[0, 1])
sns.boxplot(x="creditworthy", y="months_employed", data=data, ax=axes[0, 2])
sns.boxplot(x="creditworthy", y="avg_credit_card_bal", data=data, ax=axes[1, 0])
sns.boxplot(x="creditworthy", y="months_with_bank", data=data, ax=axes[1, 1])

axes[1, 2].axis("off")  # Hide the unused subplot

st.pyplot(fig)

# Create a new figure for the bar plot
st.write("### Bar Plot for Dummified Education Level Variables")
fig, ax = plt.subplots(figsize=(10, 5))

education_levels = [
    "education_level_Junior Certificate",
    "education_level_Leaving Certificate",
    "education_level_NFQ7",
    "education_level_NFQ8",
    "education_level_NFQ9",
]
data_melted = pd.melt(data, id_vars=["creditworthy"], value_vars=education_levels)

# Shorten the labels for the plot
data_melted["variable"] = data_melted["variable"].replace(
    {
        "education_level_Junior Certificate": "Junior",
        "education_level_Leaving Certificate": "Leaving",
        "education_level_NFQ7": "NFQ7",
        "education_level_NFQ8": "NFQ8",
        "education_level_NFQ9": "NFQ9",
    }
)

sns.barplot(x="variable", y="value", hue="creditworthy", data=data_melted, ax=ax)

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

# Split data into train and test sets
X = data.drop("creditworthy", axis=1)
y = data["creditworthy"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Show data split information and pie chart side by side
st.write("### Split Data into Train and Test")
col1, col2 = st.columns([1, 1])

with col1:
    st.write(f"Training set size: {X_train.shape[0]} samples")
    st.write(f"Test set size: {X_test.shape[0]} samples")

with col2:
    # Create pie chart for data split
    split_labels = ["Train", "Test"]
    split_sizes = [X_train.shape[0], X_test.shape[0]]
    fig, ax = plt.subplots(
        figsize=(2, 2)
    )  # Adjust figure size to make pie chart smaller
    ax.pie(
        split_sizes,
        labels=split_labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#ff9999", "#66b3ff"],
    )
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

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

# Predict and evaluate on training data
y_train_pred = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(
    y_train, y_train_pred, pos_label=1
)  # Precision for positive class
train_recall = recall_score(
    y_train, y_train_pred, pos_label=1
)  # Recall for positive class

# Predict and evaluate on testing data
y_test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(
    y_test, y_test_pred, pos_label=1
)  # Precision for positive class
test_recall = recall_score(
    y_test, y_test_pred, pos_label=1
)  # Recall for positive class

# Format the metrics as integers (percentages with no decimal places)
train_accuracy = int(train_accuracy * 100)
train_precision = int(train_precision * 100)
train_recall = int(train_recall * 100)
test_accuracy = int(test_accuracy * 100)
test_precision = int(test_precision * 100)
test_recall = int(test_recall * 100)

# Display the metrics in a table
st.write("### Model Performance")
performance_data = {
    "Set": ["Train", "Test"],
    "Accuracy": [f"{train_accuracy}%", f"{test_accuracy}%"],
    "Precision": [f"{train_precision}%", f"{test_precision}%"],
    "Recall": [f"{train_recall}%", f"{test_recall}%"],
}
performance_df = pd.DataFrame(performance_data)
st.table(performance_df)

st.write("### Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Plot confusion matrix
st.write("### Confusion Matrix")
fig, ax = plt.subplots(
    figsize=(2, 2)
)  # Adjust figure size to make confusion matrix smaller
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="coolwarm",
    xticklabels=["Not Creditworthy", "Creditworthy"],
    yticklabels=["Not Creditworthy", "Creditworthy"],
    ax=ax,
    cbar=False,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

# Adjust font size for tick labels
ax.tick_params(axis="both", which="major", labelsize=6)

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

# # Visualize the confidence
# st.write("### Prediction Confidence")
# fig, ax = plt.subplots(figsize=(5, 3))
# sns.barplot(
#     x=["Not Creditworthy", "Creditworthy"],
#     y=prediction_proba,
#     palette="coolwarm",
#     ax=ax,
# )
# ax.set_ylabel("Probability")
# st.pyplot(fig)


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
