"""
Author: [M.fadakar]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import statsmodels.formula.api as smf
from sklearn.preprocessing import RobustScaler
import random
import warnings


def signed_log1p(x: float) -> float:
    """
    Apply signed log1p transformation.
    """
    if pd.isna(x):
        return np.nan
    return np.sign(x) * np.log1p(np.abs(x))


def load_and_convert(file_path: str) -> pd.DataFrame:
    """
    Load data from Excel file.
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error while loading file: {e}")
        return pd.DataFrame()


def detect_outliers(series: pd.Series) -> tuple:
    """
    Detect outliers using the IQR method.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    is_outlier = ((series < lower_bound) | (series > upper_bound)) & (~series.isnull())
    return is_outlier, lower_bound, upper_bound


def quantile_regression_replace(train_series: pd.Series, test_series: pd.Series, quantile: float = 0.5) -> tuple:
    """
    Replace outliers in the training series using quantile regression.
    If regression fails, fall back to the median.
    Apply the same treatment to the test series based on training data bounds.
    """
    train_series_processed = train_series.copy()
    test_series_processed = test_series.copy()

    train_series_not_nan = train_series_processed.dropna()
    if train_series_not_nan.empty:
        return train_series_processed, test_series_processed

    is_outlier_train_on_not_nan, lower_bound, upper_bound = detect_outliers(train_series_not_nan)

    is_outlier_train = pd.Series(False, index=train_series_processed.index)
    if not is_outlier_train_on_not_nan.empty:
        is_outlier_train[train_series_not_nan[is_outlier_train_on_not_nan].index] = True

    non_outlier_train_median = train_series_not_nan[~is_outlier_train_on_not_nan].median()
    if pd.isna(non_outlier_train_median) and not train_series_not_nan.empty:
        non_outlier_train_median = train_series_not_nan.median()
    if pd.isna(non_outlier_train_median):
        non_outlier_train_median = 0

    replacement_value_train = non_outlier_train_median

    if not is_outlier_train.any() or len(train_series_not_nan[~is_outlier_train_on_not_nan]) < 5:
        train_series_processed[is_outlier_train] = non_outlier_train_median
    else:
        train_temp_df_not_nan = pd.DataFrame({
            'y': train_series_not_nan,
            'index': np.arange(len(train_series_not_nan))
        })
        non_outlier_data_train_for_reg = train_temp_df_not_nan[~is_outlier_train_on_not_nan]

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                if non_outlier_data_train_for_reg['index'].nunique() > 1 and len(non_outlier_data_train_for_reg) > 1:
                    model = smf.quantreg('y ~ index', non_outlier_data_train_for_reg).fit(q=quantile, max_iter=1000)

                    full_pred_temp_df = pd.DataFrame({'index': np.arange(len(train_series_processed))})
                    predictions_full = model.predict(full_pred_temp_df[['index']])

                    train_series_processed[is_outlier_train] = predictions_full[is_outlier_train.values]
                else:
                    train_series_processed[is_outlier_train] = non_outlier_train_median
        except (RuntimeWarning, ValueError, Exception):
            train_series_processed[is_outlier_train] = non_outlier_train_median

    test_series_not_nan = test_series_processed.dropna()
    if not test_series_not_nan.empty:
        is_outlier_test_on_not_nan = ((test_series_not_nan < lower_bound) | (test_series_not_nan > upper_bound))

        is_outlier_test = pd.Series(False, index=test_series_processed.index)
        if not is_outlier_test_on_not_nan.empty:
            is_outlier_test[test_series_not_nan[is_outlier_test_on_not_nan].index] = True

        if is_outlier_test.any():
            test_series_processed[is_outlier_test] = non_outlier_train_median

    return train_series_processed, test_series_processed


def display_statistics(data: pd.DataFrame, title_prefix: str = ""):
    """
    Display skewness, kurtosis, and histograms for a subset of numeric columns.
    """
    numeric_columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]

    if not numeric_columns:
        print(f"No numeric columns found ({title_prefix}).")
        return

    if len(numeric_columns) > 17:
        selected_columns = random.sample(numeric_columns, 17)
    else:
        selected_columns = numeric_columns

    n_cols = min(6, len(selected_columns))
    n_rows = (len(selected_columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), dpi=150, squeeze=False)
    axes_flat = axes.flatten()

    for idx, col_name in enumerate(selected_columns):
        ax = axes_flat[idx]
        series_data = data[col_name].dropna()

        if series_data.empty:
            ax.set_title(f'{col_name}\n(No Data)', fontsize=8)
            ax.grid(False)
            continue

        skewness = skew(series_data)
        kurt = kurtosis(series_data)

        ax.hist(series_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'{col_name}\nSk={skewness:.2f}, Ku={kurt:.2f}', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=6)
        ax.tick_params(axis='both', labelsize=6)
        ax.axvline(series_data.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean ({series_data.mean():.2f})')
        ax.axvline(series_data.median(), color='green', linestyle='dashed', linewidth=1, label=f'Median ({series_data.median():.2f})')
        ax.legend(fontsize=6)
        ax.grid(False)

    for idx in range(len(selected_columns), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    fig.suptitle(f"Data Distribution ({title_prefix})", fontsize=12, y=1.02 if n_rows > 0 else 1.1)
    plt.tight_layout(rect=[0, 0, 1, 0.98 if n_rows > 0 else 1])
    plt.show()


def main():
    """
    Main function: data preprocessing with train-test split, 
    signed log1p transformation, outlier handling, and RobustScaler normalization.
    """
    data_full = load_and_convert('cleaned_All  Data.xlsx')
    if data_full.empty:
        print("No data loaded. Exiting program.")
        return

    total_rows = len(data_full)
    train_size = int(0.7 * total_rows)

    train_df_full = data_full.iloc[:train_size].copy()
    test_df_full = data_full.iloc[train_size:].copy()

    numeric_cols = train_df_full.select_dtypes(include=np.number).columns.tolist()
    non_numeric_cols = train_df_full.select_dtypes(exclude=np.number).columns.tolist()

    train_data_numeric = train_df_full[numeric_cols].copy()
    test_data_numeric = test_df_full[numeric_cols].copy()

    train_data_non_numeric = train_df_full[non_numeric_cols].reset_index(drop=True)
    test_data_non_numeric = test_df_full[non_numeric_cols].reset_index(drop=True)

    for col in train_data_numeric.columns:
        train_data_numeric[col] = train_data_numeric[col].apply(signed_log1p)
        if col in test_data_numeric:
            test_data_numeric[col] = test_data_numeric[col].apply(signed_log1p)

    train_data_log_transformed = train_data_numeric
    test_data_log_transformed = test_data_numeric

    constant_cols = []
    for col in train_data_log_transformed.columns:
        unique_values = train_data_log_transformed[col].dropna().unique()
        if len(unique_values) == 1 or (len(unique_values) == 0 and train_data_log_transformed[col].isnull().all()):
            constant_cols.append(col)

    if constant_cols:
        print(f"Constant (or all-NaN) columns identified: {constant_cols}")

    train_data_constant = train_data_log_transformed[constant_cols].reset_index(drop=True)
    test_data_constant_cols = [col for col in constant_cols if col in test_data_log_transformed.columns]
    test_data_constant = test_data_log_transformed[test_data_constant_cols].reset_index(drop=True)

    train_data_for_scaling = train_data_log_transformed.drop(columns=constant_cols)
    test_data_for_scaling = test_data_log_transformed.drop(columns=test_data_constant_cols)

    if not train_data_for_scaling.empty:
        for col in train_data_for_scaling.columns:
            if col in test_data_for_scaling.columns:
                train_data_for_scaling[col], test_data_for_scaling[col] = quantile_regression_replace(
                    train_data_for_scaling[col], test_data_for_scaling[col], quantile=0.75
                )
            else:
                train_data_for_scaling[col], _ = quantile_regression_replace(
                    train_data_for_scaling[col], pd.Series(dtype='float64'), quantile=0.75
                )

    train_data_scaled_df = pd.DataFrame(index=train_data_for_scaling.index)
    test_data_scaled_df = pd.DataFrame(index=test_data_for_scaling.index)

    if not train_data_for_scaling.empty:
        scaler = RobustScaler()
        scaler.fit(train_data_for_scaling)

        train_data_scaled_np = scaler.transform(train_data_for_scaling)
        train_data_scaled_df = pd.DataFrame(train_data_scaled_np, columns=train_data_for_scaling.columns, index=train_data_for_scaling.index).reset_index(drop=True)

        if not test_data_for_scaling.empty:
            test_data_scaled_np = scaler.transform(test_data_for_scaling)
            test_data_scaled_df = pd.DataFrame(test_data_scaled_np, columns=test_data_for_scaling.columns, index=test_data_for_scaling.index).reset_index(drop=True)
        else:
            test_data_scaled_df = pd.DataFrame(columns=train_data_for_scaling.columns)
    else:
        train_data_scaled_df = pd.DataFrame(columns=train_data_log_transformed.columns).reset_index(drop=True)
        test_data_scaled_df = pd.DataFrame(columns=test_data_log_transformed.columns).reset_index(drop=True)

    train_final_numeric = pd.concat([train_data_scaled_df, train_data_constant], axis=1)
    test_final_numeric = pd.concat([test_data_scaled_df, test_data_constant], axis=1)

    train_final_numeric = train_final_numeric[numeric_cols]
    test_final_numeric = test_final_numeric[numeric_cols]

    train_output_df = pd.concat([train_final_numeric, train_data_non_numeric], axis=1)
    test_output_df = pd.concat([test_final_numeric, test_data_non_numeric], axis=1)

    train_output_df = train_output_df[data_full.columns.intersection(train_output_df.columns)]
    test_output_df = test_output_df[data_full.columns.intersection(test_output_df.columns)]

    try:
        train_output_df.to_excel('Alldata_train.xlsx', index=False)
        test_output_df.to_excel('Alldata_test.xlsx', index=False)
        print("\nFiles Alldata_train.xlsx and Alldata_test.xlsx successfully saved.")
    except Exception as e:
        print(f"Error saving Excel files: {e}")

    print("\nTraining data statistics (final - after RobustScaler):")
    display_statistics(train_output_df, title_prefix="Train Final (Post-RobustScaler)")
    print("\nTest data statistics (final - after RobustScaler):")
    display_statistics(test_output_df, title_prefix="Test Final (Post-RobustScaler)")


if __name__ == "__main__":
    main()
