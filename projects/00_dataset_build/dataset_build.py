import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples and features
num_samples = 2000
num_features = 20

# Generate features with normal distribution
X = np.random.randn(num_samples, num_features)

# Introduce high noise in three selected features
noise_level = 5  # Higher noise level
high_noise_indices = np.random.choice(num_features, 3, replace=False)
for idx in high_noise_indices:
    X[:, idx] += np.random.randn(num_samples) * noise_level

# Ensure 8 features are strongly correlated with the target but not with themselves
correlated_features = np.random.choice(num_features, 8, replace=False)
true_coefficients = np.zeros(num_features)
true_coefficients[correlated_features] = np.random.uniform(40, 90, 8)  # Stronger influence

# Generate independent noise to reduce multicollinearity
independent_noise = np.random.randn(num_samples, 8) * 0.3  # Small perturbations
X[:, correlated_features] += independent_noise  # Introduce variations to reduce correlation among selected features

# Generate target variable as a weighted sum of selected features with additional noise
y = (X[:, correlated_features] @ true_coefficients[correlated_features]) + np.random.randn(num_samples) * 1.5  # Lower noise

# Create DataFrame
df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(num_features)])
df['Target'] = y

# Save dataset to CSV (optional)
df.to_csv("synthetic_regression_dataset.csv", index=False)

# Display first few rows
print(df.head())
