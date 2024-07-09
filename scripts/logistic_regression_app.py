import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2


# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)


# Load and preprocess data
data = load_data("data/fraud_detection.csv")

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
data = pd.get_dummies(data, columns=["transaction_type"])

# Show the transformed data
st.write("### Transformed Dataset with Dummified Variables")
st.write(data.head())

# Visualizations
st.write("### Data Visualizations")
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

sns.boxplot(x="is_fraud", y="trans_amount", data=data, ax=axes[0, 0])
sns.boxplot(x="is_fraud", y="trans_max_ratio", data=data, ax=axes[0, 1])
sns.boxplot(x="is_fraud", y="trans_daily_count", data=data, ax=axes[0, 2])
sns.boxplot(x="is_fraud", y="hourly_prob", data=data, ax=axes[1, 0])
sns.boxplot(x="is_fraud", y="abroad", data=data, ax=axes[1, 1])

# Bar plot for the dummified transaction_type variables
transaction_types = [
    "transaction_type_online",
    "transaction_type_instore",
]
data_melted = pd.melt(data, id_vars=["is_fraud"], value_vars=transaction_types)
sns.barplot(x="variable", y="value", hue="is_fraud", data=data_melted, ax=axes[1, 2])

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
Training a logistic regression model on the prepared dataset.
"""
)

X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)

st.write("### Logistic Regression Model")
st.write(f"Model Coefficients: {clf.coef_}")

# Add threshold slider to the sidebar
threshold = st.sidebar.slider(
    "Threshold for Fraud Prediction", min_value=0.0, max_value=1.0, value=0.5, step=0.01
)

# Model Evaluation
st.header("Model Evaluation")
st.markdown(
    """
Evaluating the model on a test set.
"""
)

# Predict probabilities
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Apply threshold to get predictions
y_pred = (y_pred_proba >= threshold).astype(int)

# Calculate the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)  # Precision for positive class
recall = recall_score(y_test, y_pred)  # Recall for positive class

# Display the metrics
st.write("### Model Performance")
st.metric(label="Accuracy", value=f"{accuracy:.2f}")
st.metric(label="Precision (Positive Class)", value=f"{precision:.2f}")
st.metric(label="Recall (Positive Class)", value=f"{recall:.2f}")

# Explanation of precision and recall
st.markdown(
    """
### Understanding Precision and Recall

- **Precision**: Of all the transactions predicted as fraud, how many were actually fraud? High precision means fewer false positives.
- **Recall**: Of all the actual fraud transactions, how many were correctly predicted as fraud? High recall means fewer false negatives.

Even with high accuracy, a low recall indicates that many fraud transactions are missed.
"""
)

# Explanation of precision and recall
st.markdown(
    """
### Still a bit confusing? Think of it in business terms

##### What does a model with high precision and low recall look like in a business context?
- We are low on false positives or false alarms. It means that customers are rarely bothered with incorrect fraud alerts.
- However, we are missing a lot of fraud transactions. This could lead to significant financial losses for the bank or customer.

##### What does a model with low precision and high recall look like in a business context?
- We are catching a lot of fraud transactions. This is great for preventing financial losses.
- However, we are also raising a lot of false alarms. This could lead to customer dissatisfaction and loss of trust in the bank.

- So we have to balance the two.  In practise, catching fraud is probably enough to justify a few false alarms. We find the right balance by adjusting the threshold.
"""
)

# Visualize precision and recall impact
st.write("### Precision and Recall Impact")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()


# Plot confusion matrix
st.write("### Confusion Matrix")
fig, ax = plt.subplots(figsize=(3, 3))  # Adjust the size here
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="coolwarm",
    xticklabels=["Not Fraud", "Fraud"],
    yticklabels=["Not Fraud", "Fraud"],
    ax=ax,
    cbar=False,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

# Move the x-axis label to the top
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()

st.pyplot(fig)

# Explanation of confusion matrix
st.markdown(
    """
### Confusion Matrix
- **True Positives (TP)**: Correctly predicted fraud transactions.
- **True Negatives (TN)**: Correctly predicted non-fraud transactions.
- **False Positives (FP)**: Incorrectly predicted fraud transactions (actually not fraud).
- **False Negatives (FN)**: Incorrectly predicted non-fraud transactions (actually fraud).

A high number of false negatives (FN) indicates a low recall.
"""
)


# Function to calculate precision and recall for various thresholds
def calculate_precision_recall(clf, X_test, y_test, thresholds):
    precision_scores = []
    recall_scores = []

    for threshold in thresholds:
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))

    return precision_scores, recall_scores


# Define thresholds
thresholds = np.arange(0, 1.00, 0.05)

# Calculate precision and recall
precision_scores, recall_scores = calculate_precision_recall(
    clf, X_test, y_test, thresholds
)

# Plot the relationship between threshold and precision/recall
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, precision_scores, label="Precision", marker="o")
ax.plot(thresholds, recall_scores, label="Recall", marker="o")
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision and Recall vs. Threshold")
ax.legend(loc="best")

st.pyplot(fig)
