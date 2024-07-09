import numpy as np
import pandas as pd

np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate features
age = np.random.randint(20, 60, size=n_samples)
income = np.random.randint(30000, 120000, size=n_samples)
months_employed = np.random.randint(0, 240, size=n_samples)
education_level = np.random.choice(
    ["Junior Certificate", "Leaving Certificate", "NFQ7", "NFQ8", "NFQ9"],
    size=n_samples,
)
avg_credit_card_bal = np.random.randint(0, 15000, size=n_samples)
months_with_bank = np.random.randint(0, 240, size=n_samples)

# Generate target variable with more complex relationships
creditworthy = (
    (income > 70000).astype(int)
    + ((income > 50000) & (months_employed > 50)).astype(int)
    + ((avg_credit_card_bal < 3000) | (avg_credit_card_bal < 1000)).astype(int)
    + (education_level == "Leaving Certificate").astype(int)
    + (education_level == "NFQ7").astype(int)
    + ((age > 30) & (months_with_bank > 50)).astype(int)
    + ((age >= 30) & (months_with_bank < 50)).astype(int)
)

# Add some noise (reduced noise)
creditworthy = (
    creditworthy + np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
) > 2
creditworthy = creditworthy.astype(int)

# Balance the dataset
n_positive = np.sum(creditworthy)
n_negative = n_samples - n_positive

if n_positive > n_negative:
    idx_positive = np.where(creditworthy == 1)[0]
    idx_to_remove = np.random.choice(
        idx_positive, size=n_positive - n_negative, replace=False
    )
else:
    idx_negative = np.where(creditworthy == 0)[0]
    idx_to_remove = np.random.choice(
        idx_negative, size=n_negative - n_positive, replace=False
    )

mask = np.ones(n_samples, dtype=bool)
mask[idx_to_remove] = False

data = pd.DataFrame(
    {
        "age": age[mask],
        "income": income[mask],
        "months_employed": months_employed[mask],
        "education_level": education_level[mask],
        "avg_credit_card_bal": avg_credit_card_bal[mask],
        "months_with_bank": months_with_bank[mask],
        "creditworthy": creditworthy[mask],
    }
)

# Dummify the categorical variable
# data = pd.get_dummies(data, columns=['education_level'], drop_first=True)

# Save the dataset to a CSV file
data.to_csv("data/credit_scoring.csv", index=False)
