"""
Script Name: Robust Preprocessing Pipeline for Financial Time Series
Description:
    This script implements a preprocessing pipeline for financial time series.
    Key steps include:
        - Loading and cleaning the dataset
        - Handling zeros and missing values
        - Outlier treatment with quantile regression
        - Scaling using RobustScaler (with saving and loading support)
        - Generating descriptive statistics and visualizations
        - Chronological train-test split
    Output files:
        - Preprocessed train and test sets (scaled)
        - Saved RobustScaler object
        - Optionally, inverse-transformed data
"""

import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import statsmodels.formula.api as smf
from sklearn.preprocessing import RobustScaler


# ------------------------------
# Data Loading
# ------------------------------
def load_and_convert(file_path: str) -> pd.DataFrame:
    """Load data from XLSX file, removing Date column if it exists."""
    try:
        df = pd.read_excel(file_path)
        if 'Date' in df.columns:
            df = df.drop(columns=['Date'])
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error while loading file: {e}")
        return pd.DataFrame()


# ------------------------------
# Handling Missing Values and Zeros
# ------------------------------
def replace_zeros_and_nans(df: pd.DataFrame, replacement_values: dict = None) -> tuple:
    """
    Replace zeros and NaNs with appropriate values, excluding Date column.
    If replacement_values is None, calculate from df; otherwise, use provided values.
    Returns: (processed_df, replacement_values)
    """
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col.lower() != 'date']

    if replacement_values is None:
        replacement_values = {}
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            non_zero_values = df[column][(df[column] != 0) & (~df[column].isnull())]
            if len(non_zero_values) > 0:
                replacement_values[column] = non_zero_values.median()
            else:
                replacement_values[column] = 1e-4
            df[column] = df[column].replace({0: replacement_values[column], np.nan: replacement_values[column]})
    else:
        for column in numeric_columns:
            if column in replacement_values:
                df[column] = df[column].replace({0: replacement_values[column], np.nan: replacement_values[column]})

    return df, replacement_values


# ------------------------------
# Outlier Detection and Treatment
# ------------------------------
def detect_outliers(series: pd.Series) -> tuple:
    """Detect outliers using the IQR method."""
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    is_outlier = (series < lower_bound) | (series > upper_bound)
    return is_outlier, lower_bound, upper_bound


def quantile_regression_replace(train_series: pd.Series, test_series: pd.Series, quantile: float = 0.5) -> tuple:
    """
    Replace outliers in train series using quantile regression trained on train data.
    Test series remains unchanged.
    """
    train_series = train_series.copy()
    test_series = test_series.copy()

    # Detect outliers in train data
    is_outlier_train, _, _ = detect_outliers(train_series)

    # Prepare regression DataFrame
    train_temp_df = pd.DataFrame({
        'y': train_series,
        'index': np.arange(len(train_series))
    })

    # Train quantile regression
    non_outlier_data = train_temp_df[~is_outlier_train]
    if len(non_outlier_data) > 1:
        model = smf.quantreg('y ~ index', non_outlier_data).fit(q=quantile)
        train_predictions = model.predict(train_temp_df[['index']])
        train_series[is_outlier_train] = train_predictions[is_outlier_train]
    else:
        median_value = train_series[~is_outlier_train].median()
        train_series[is_outlier_train] = median_value

    return train_series, test_series


# ------------------------------
# Statistics and Visualization
# ------------------------------
def display_statistics(data: pd.DataFrame):
    """Display skewness, kurtosis, and distribution plots for selected numeric columns."""
    numeric_columns = [col for col in data.select_dtypes(include=[np.number]).columns if col.lower() != 'date']
    selected_columns = random.sample(numeric_columns, min(len(numeric_columns), 17))

    for col in selected_columns:
        print(f"\n📊 Statistics for {col}:")
        skewness = skew(data[col])
        kurt = kurtosis(data[col])
        print(f"  Skewness: {skewness:.4f}")
        print(f"  Kurtosis: {kurt:.4f}")

        plt.figure(figsize=(12, 8), dpi=300)
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.hist(data[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.axvline(data[col].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean')
        plt.axvline(data[col].median(), color='green', linestyle='dashed', linewidth=1, label='Median')
        plt.legend()
        plt.tight_layout()
        plt.show()


# ------------------------------
# Main Preprocessing Workflow
# ------------------------------
def main():
    """Main pipeline: preprocessing, scaling, saving outputs, and visualizations."""
    # Load data
    data = load_and_convert("cleaned_lreturns30.xlsx")
    if data.empty:
        print("❌ No data loaded. Exiting.")
        return

    # Ensure chronological order
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        dates = data['Date']
        data_without_date = data.drop(columns=['Date'])
    else:
        print("⚠️ Date column not found. Using index as time order.")
        dates = pd.Series(range(len(data)))
        data_without_date = data.copy()

    # Train-test split (70%-30%)
    total_rows = len(data_without_date)
    train_size = int(0.7 * total_rows)
    train_data = data_without_date.iloc[:train_size].copy()
    test_data = data_without_date.iloc[train_size:].copy()
    train_dates, test_dates = dates.iloc[:train_size], dates.iloc[train_size:]

    # Handle missing values
    train_data_replaced, replacement_values = replace_zeros_and_nans(train_data.copy())
    test_data_replaced, _ = replace_zeros_and_nans(test_data.copy(), replacement_values)

    # Outlier treatment
    for col in train_data_replaced.columns:
        if train_data_replaced[col].dtype in ['int64', 'float64']:
            train_data_replaced[col], test_data_replaced[col] = quantile_regression_replace(
                train_data_replaced[col], test_data_replaced[col], quantile=0.5
            )

    # Scaling with RobustScaler
    scaler = RobustScaler()
    scaler.fit(train_data_replaced)

    train_scaled = scaler.transform(train_data_replaced)
    test_scaled = scaler.transform(test_data_replaced)

    train_scaled_df = pd.DataFrame(train_scaled, columns=train_data_replaced.columns)
    test_scaled_df = pd.DataFrame(test_scaled, columns=test_data_replaced.columns)

    # Add Date back
    train_scaled_df['Date'] = train_dates.values
    test_scaled_df['Date'] = test_dates.values

    # Save scaler
    with open("robust_scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)
    print("✅ RobustScaler saved to 'robust_scaler.pkl'.")

    # Save outputs
    train_scaled_df.to_excel("lreturns30w_train_scaled.xlsx", index=False, float_format="%.7f")
    test_scaled_df.to_excel("lreturns30w_test_scaled.xlsx", index=False, float_format="%.7f")

    # Display statistics
    print("\nTrain Data Statistics (after RobustScaler):")
    display_statistics(train_scaled_df)
    print("\nTest Data Statistics (after RobustScaler):")
    display_statistics(test_scaled_df)


# ------------------------------
# Inverse Transformation
# ------------------------------
def inverse_robust_scale(data: pd.DataFrame, scaler_path: str = "robust_scaler.pkl") -> pd.DataFrame:
    """Inverse transform scaled data using saved RobustScaler."""
    try:
        with open(scaler_path, "rb") as file:
            loaded_scaler = pickle.load(file)
        scaled_values = data.drop(columns=['Date'], errors='ignore').values
        inverse_transformed_values = loaded_scaler.inverse_transform(scaled_values)
        inverse_transformed_df = pd.DataFrame(inverse_transformed_values, columns=data.drop(columns=['Date'], errors='ignore').columns)
        if 'Date' in data.columns:
            inverse_transformed_df['Date'] = data['Date'].values
        return inverse_transformed_df
    except Exception as e:
        print(f"❌ Error during inverse transform: {e}")
        return pd.DataFrame()


# ------------------------------
# Run Script
# ------------------------------
if __name__ == "__main__":
    main()

    # Example: inverse transformation
    train_scaled = pd.read_excel("lreturns30w_train_scaled.xlsx")
    test_scaled = pd.read_excel("lreturns30w_test_scaled.xlsx")

    train_original = inverse_robust_scale(train_scaled)
    test_original = inverse_robust_scale(test_scaled)

    train_original.to_excel("lreturns30w_train_original.xlsx", index=False, float_format="%.7f")
    test_original.to_excel("lreturns30w_test_original.xlsx", index=False, float_format="%.7f")

    print("\n✅ Inverse transformation example completed.")
