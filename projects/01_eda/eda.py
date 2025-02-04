import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
from statsmodels.tools import add_constant

# Load dataset
df = pd.read_csv("synthetic_regression_dataset.csv")

# Basic info and summary
print("Dataset Info:")
df.info()
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Ensure output directory exists
output_dir = "eda_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Compute correlation with target
correlation_with_target = df.corr()["Target"].drop("Target").abs()
top_8_features = correlation_with_target.nlargest(8)
print("\nTop 8 highly correlated features with Target:")
print(top_8_features)

# Exploratory Data Analysis (EDA)
# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()

# Pairplot of a subset of features
sns.pairplot(df.iloc[:, :5])  # First 5 features
plt.savefig(os.path.join(output_dir, "pairplot.png"))
plt.close()

# Distribution of target variable
plt.figure(figsize=(8, 5))
sns.histplot(df['Target'], bins=30, kde=True)
plt.title("Distribution of Target Variable")
plt.savefig(os.path.join(output_dir, "target_distribution.png"))
plt.close()

# Boxplots for features with high noise
plt.figure(figsize=(10, 6))
noisy_features = [f"Feature_{i+1}" for i in [0, 5, 10]]  # Example selection
sns.boxplot(data=df[noisy_features])
plt.title("Boxplot of Noisy Features")
plt.savefig(os.path.join(output_dir, "noisy_features_boxplot.png"))
plt.close()

print("EDA complete. Plots saved in 'eda_visualizations' directory.")

# Function to transform features
def transform_features(df):
    transformed_df = df.copy()
    for col in df.columns[:-1]:  # Exclude target column
        transformed_df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))  # Log transform
        transformed_df[f"{col}_sqrt"] = np.sqrt(df[col].clip(lower=0))  # Square root transform
        transformed_df[f"{col}_log_sqrt"] = np.log1p(np.sqrt(df[col].clip(lower=0)))  # Log of sqrt
    return transformed_df

# Apply transformations
df_transformed = transform_features(df)

# Display first few rows
print(df_transformed.head())

# Function to calculate Variance Inflation Factor (VIF)
def calculate_vif(df):
    df_with_const = add_constant(df)  # Add an intercept column
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(df_with_const.values, i) for i in range(df_with_const.shape[1])]
    return vif_data[vif_data["Feature"] != "const"]  # Exclude the constant

# Calculate VIF excluding target variable
vif_results = calculate_vif(df.iloc[:, :-1])
print("\nVariance Inflation Factor (VIF) values:")
print(vif_results)

# Function to check VIF threshold
def check_high_vif(df, threshold=3):
    vif_data = calculate_vif(df)
    high_vif_features = vif_data[vif_data["VIF"] > threshold]
    if not high_vif_features.empty:
        print("\nFeatures with VIF higher than", threshold, ":")
        print(high_vif_features)
    else:
        print("\nNo features exceed the VIF threshold.")
    return high_vif_features

# Display first few rows
print(df_transformed.head())

# Function to detect and replace outliers
def detect_and_replace_outliers(df, method="iqr", threshold=1.5, strategy="median"):
    df_cleaned = df.copy()
    outliers_dict = {}

    for col in df.select_dtypes(include=[np.number]).columns:  # Only check numeric columns
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        elif method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            z_scores = (df[col] - mean) / std
            outliers = df[np.abs(z_scores) > threshold].index
        else:
            raise ValueError("Invalid method. Use 'iqr' or 'zscore'.")

        outliers_dict[col] = list(outliers)

        if strategy == "mean":
            replacement_value = df[col].mean()
        elif strategy == "median":
            replacement_value = df[col].median()
        elif strategy == "mode":
            replacement_value = df[col].mode()[0]
        elif strategy == "remove":
            df_cleaned = df_cleaned.drop(outliers)
            continue
        elif strategy == "iqr_bound":
            for index in outliers:
                if df.loc[index, col] < lower_bound:
                    df_cleaned.loc[index, col] = lower_bound
                else:
                    df_cleaned.loc[index, col] = upper_bound
        else:
            raise ValueError("Invalid strategy. Use 'mean', 'median', 'mode', 'remove', or 'iqr_bound'.")

    return df_cleaned, outliers_dict

    for col in df.select_dtypes(include=[np.number]).columns:  # Only check numeric columns
        if method == "iqr":
            Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
            Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index

        elif method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            z_scores = (df[col] - mean) / std
            outliers = df[np.abs(z_scores) > threshold].index

        else:
            raise ValueError("Invalid method. Use 'iqr' or 'zscore'.")

        outliers_dict[col] = list(outliers)

    return outliers_dict


# Detect and replace outliers
df_cleaned, outliers = detect_and_replace_outliers(df, method="iqr", strategy="iqr_bound")


def detect_outliers(df, method="iqr", threshold=1.5):
    """
    Detects outliers in each numerical column of a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - method (str): The method to detect outliers. Options: "iqr" (default), "zscore".
    - threshold (float): The threshold for detecting outliers.
        - IQR default = 1.5
        - Z-score default = 3

    Returns:
    - outliers_dict (dict): A dictionary where keys are column names and values are lists of outlier indices.
    """
    outliers_dict = {}

    for col in df.select_dtypes(include=[np.number]).columns:  # Only check numeric columns
        if method == "iqr":
            Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
            Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index

        elif method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            z_scores = (df[col] - mean) / std
            outliers = df[np.abs(z_scores) > threshold].index

        else:
            raise ValueError("Invalid method. Use 'iqr' or 'zscore'.")

        outliers_dict[col] = list(outliers)

    return outliers_dict


# Example usage
outliers = detect_outliers(df_cleaned, method="iqr")
print("Outliers detected:", outliers)

# Confirmed 8 features selected
cols_to_keep = pd.Series(top_8_features.index)
trimmed_dataset = df_cleaned.loc[:, cols_to_keep]
trimmed_dataset['Target'] = df_cleaned['Target']

# Save dataset to CSV (optional)
trimmed_dataset.to_csv("cleaned_synthetic_regression_dataset.csv", index=False)