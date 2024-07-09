import numpy as np
import pandas as pd

np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate features
trans_amount = np.random.randint(10, 10000, size=n_samples)
trans_max_ratio = np.random.rand(
    n_samples
)  # Ratio to the maximum transaction amount for the account
transaction_type = np.random.choice(["online", "instore"], size=n_samples)
trans_daily_count = np.random.randint(1, 20, size=n_samples)
abroad = np.random.choice([0, 1], size=n_samples)
hourly_prob = np.random.rand(n_samples)  # Probability of transaction at that hour
trusted_counterparty = np.random.choice([True, False], size=n_samples)

# Generate target variable with complex relationships
is_fraud = (
    (trans_amount > 5000).astype(int)
    + ((trans_amount > 3000) & (trans_daily_count > 10)).astype(int)
    + ((transaction_type == "online") & (trans_amount > 2000)).astype(int)
    + (abroad == 1).astype(int)
    + ((trusted_counterparty == False) & (trans_amount > 1000)).astype(int)
    + (hourly_prob < 0.2).astype(int)
    + (trans_max_ratio > 0.8).astype(int)
)

# Add some noise to make it a bit more realistic
is_fraud = (is_fraud + np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])) > 2
is_fraud = is_fraud.astype(int)

# Adjust to make only 10% fraud
n_positive = np.sum(is_fraud)
desired_positive = int(0.1 * n_samples)
desired_negative = n_samples - desired_positive

if n_positive > desired_positive:
    idx_positive = np.where(is_fraud == 1)[0]
    idx_to_remove = np.random.choice(
        idx_positive, size=n_positive - desired_positive, replace=False
    )
    mask = np.ones(n_samples, dtype=bool)
    mask[idx_to_remove] = False
elif n_positive < desired_positive:
    idx_negative = np.where(is_fraud == 0)[0]
    idx_to_remove = np.random.choice(
        idx_negative, size=desired_negative - n_negative, replace=False
    )
    mask = np.ones(n_samples, dtype=bool)
    mask[idx_to_remove] = False
else:
    mask = np.ones(n_samples, dtype=bool)

data = pd.DataFrame(
    {
        "trans_amount": trans_amount[mask],
        "trans_max_ratio": trans_max_ratio[mask],
        "transaction_type": transaction_type[mask],
        "trans_daily_count": trans_daily_count[mask],
        "abroad": abroad[mask],
        "hourly_prob": hourly_prob[mask],
        "trusted_counterparty": trusted_counterparty[mask],
        "is_fraud": is_fraud[mask],
    }
)

# Save the dataset to a CSV file
data.to_csv("data/fraud_detection.csv", index=False)
