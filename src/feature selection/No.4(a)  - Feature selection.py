# -*- coding: utf-8 -*-
"""
Feature Selection and Model Training Script
-------------------------------------------
This script performs:
1. Data loading and cleaning
2. Feature selection with Spearman correlation
3. Hyperparameter tuning with Bayesian Optimization
4. XGBoost feature filtering and correlation management
5. Model training, cross-validation, and test evaluation
6. SHAP analysis for interpretability
7. Leakage analysis
8. Saving final selected features and results

Author: [M.fadakar]
"""

import os
import sys
import glob
import time
import random
import pickle
import logging

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from skopt.callbacks import DeltaYStopper


# -------------------------------------------------------------------
# Initial Settings
# -------------------------------------------------------------------
np.random.seed(42)
random.seed(42)

os.environ['PYTHONIOENCODING'] = 'utf-8'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_selection_log.txt'),
        logging.StreamHandler()
    ]
)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10
})

# -------------------------------------------------------------------
# Configuration Parameters
# -------------------------------------------------------------------
CONFIG = {
    'spearman_percentile': 60,
    'bayes_n_iter': 30,
    'version': 'v8_separate_train_test',
    'shap_sample_size': 100,
    'early_stopping_delta': 0.005,
    'early_stopping_patience': 20,
    'k_folds': 5,
    'correlation_threshold': 0.9,
    'leakage_importance_threshold': 0.5
}

# -------------------------------------------------------------------
# Define File Paths
# -------------------------------------------------------------------
BASE_DIR = 'feature_selection_results'
os.makedirs(BASE_DIR, exist_ok=True)


def get_path(filename):
    return os.path.join(BASE_DIR, f'{filename}_{CONFIG["version"]}')


SPEARMAN_FILE = get_path('spearman_selected_features_train') + '.pkl'
XGB_FILTERED_FEATURES_FILE = get_path('xgb_filtered_features_train') + '.pkl'
XGB_FILTERED_FEATURES_CSV_FILE = get_path('xgb_filtered_features_list_train') + '.csv'
BAYES_OPT_FILE = get_path('bayes_opt_results_train') + '.pkl'
SHAP_IMPORTANCE_FILE = get_path('shap_importance_train') + '.csv'
XGBOOST_IMPORTANCE_PLOT_FILE = get_path('xgboost_importance_plot_train') + '.png'
SHAP_IMPORTANCE_PLOT_FILE = get_path('shap_importance_plot_train') + '.png'
SHAP_SUMMARY_PLOT_FILE = get_path('shap_summary_plot_train') + '.png'
CORRELATION_HEATMAP_SPEARMAN_FILE = get_path('correlation_heatmap_spearman_train') + '.png'
CORRELATION_HEATMAP_FINAL_FILE = get_path('correlation_heatmap_final_train') + '.png'
FINAL_FEATURES_FILE = get_path('important_features_train') + '.csv'
FINAL_FEATURES_LIST_FILE = get_path('important_features_list_train') + '.csv'
CV_RESULTS_FILE = get_path('cv_results_train') + '.csv'
ACTUAL_VS_PREDICTED_PLOT_FILE = get_path('actual_vs_predicted_test') + '.png'
DATA_STORAGE_FILE = get_path('data_storage_train') + '.pkl'
LEAKAGE_ANALYSIS_FILE = get_path('leakage_analysis_train') + '.csv'
TEST_EVALUATION_FILE = get_path('test_evaluation') + '.txt'

# -------------------------------------------------------------------
# --- Clear Previous Files ---
# -------------------------------------------------------------------

def clear_previous_files():
    logging.info("Deleting previous output files...")
    patterns = [
        os.path.join(BASE_DIR, f'*{CONFIG["version"]}*'),
        os.path.join(BASE_DIR, '*v6_xgb_mdape_leakage*'),
        os.path.join(BASE_DIR, '*v5_xgb_reset*'),
        os.path.join(BASE_DIR, '*v4_xgb_filter_fix*')
    ]
    for pattern in patterns:
        files_to_delete = glob.glob(pattern)
        if not files_to_delete:
            continue
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")
                
# -------------------------------------------------------------------
# --- Helper Functions ---
# -------------------------------------------------------------------

def spearman_feature_selection(X, y, percentile):
    if X.empty or y.empty:
        logging.error("Input data to spearman_feature_selection is empty.")
        return []
    corrs = []
    valid_columns = []
    with tqdm(total=len(X.columns), desc="Spearman Correlation") as pbar:
        for col in X.columns:
            pbar.update(1)
            if X[col].nunique(dropna=False) > 1 and not X[col].isna().all() and not y.isna().all():
                mask = ~X[col].isna() & ~y.isna()
                if mask.sum() > 1:
                    try:
                        corr_val, _ = spearmanr(X.loc[mask, col], y[mask])
                        if not np.isnan(corr_val):
                            corrs.append(abs(corr_val))
                            valid_columns.append(col)
                    except Exception as e:
                        logging.warning(f"Error calculating Spearman for {col}: {e}")
    if not corrs:
        logging.error("No valid correlations were calculated.")
        return []
    corrs_series = pd.Series(corrs, index=valid_columns)
    threshold = np.percentile(corrs_series, percentile)
    selected = corrs_series[corrs_series >= threshold].index.tolist()
    logging.info(f"Number of features selected by Spearman: {len(selected)}")
    return selected

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = np.abs(y_true) >= 1e-9
    if np.sum(non_zero_mask) == 0:
        return np.inf
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    return mape

def median_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = np.abs(y_true) >= 1e-9
    if np.sum(non_zero_mask) == 0:
        return np.inf
    ape = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]) * 100
    mdape = np.median(ape)
    return mdape

def check_invalid_values(df, name="DataFrame"):
    has_nan = df.isna().any().any()
    has_inf = np.isinf(df.select_dtypes(include=[np.number])).any().any()
    if has_nan or has_inf:
        logging.error(f"{name} contains invalid values: NaN={has_nan}, Inf={has_inf}")
        return False
    return True

def plot_correlation_heatmap(data, title, filename):
    try:
        corr_matrix = data.corr(numeric_only=True)
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=0.9, vmin=-0.9, center=0,
                    annot=False, square=True, linewidths=.5, cbar_kws={"shrink": .75})
        plt.title(title, fontsize=14)
        tick_fontsize = max(4, 10 - len(data.columns) // 10)
        plt.xticks(rotation=90, fontsize=tick_fontsize)
        plt.yticks(rotation=0, fontsize=tick_fontsize)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Correlation heatmap saved to {filename}")
    except Exception as e:
        logging.error(f"Error plotting heatmap: {e}")

def validate_xgb_params(params):
    required_params = ['colsample_bytree', 'learning_rate', 'subsample']
    for param in required_params:
        if param not in params or params[param] <= 0 or params[param] > 1:
            return False
    if 'max_depth' not in params or params['max_depth'] < 1:
        return False
    if 'n_estimators' not in params or params['n_estimators'] < 10:
        return False
    return True

def xgboost_correlation_filter(X_train, y_train, feature_names, best_params, correlation_threshold):
    logging.info("Starting secondary filtering with XGBoost and correlation management...")
    if X_train.empty or y_train.empty or not feature_names:
        logging.error("Input data to xgboost_correlation_filter is empty.")
        return feature_names

    if not check_invalid_values(X_train, "X_train") or not check_invalid_values(y_train.to_frame(), "y_train"):
        logging.error("Input data contains NaN or Inf.")
        return feature_names

    if not validate_xgb_params(best_params):
        logging.error("Invalid XGBoost parameters.")
        return feature_names

    xgb_model = XGBRegressor(random_state=42, **best_params)
    try:
        xgb_model.fit(X_train, y_train)
    except Exception as e:
        logging.error(f"Error training XGBoost model: {e}")
        return feature_names

    importances = pd.Series(xgb_model.feature_importances_, index=feature_names)
    importances = importances[importances > 1e-8].sort_values(ascending=False)
    if importances.empty:
        logging.error("No important features found.")
        return feature_names

    current_features = importances.index.tolist()
    X_train_current = X_train[current_features]
    logging.info(f"Number of features before correlation filter: {len(current_features)}")

    corr_matrix = X_train_current.corr().abs()
    features_to_drop = set()
    for i in range(len(current_features)):
        for j in range(i + 1, len(current_features)):
            f1, f2 = current_features[i], current_features[j]
            if f1 in features_to_drop or f2 in features_to_drop:
                continue
            if corr_matrix.loc[f1, f2] >= correlation_threshold:
                if importances[f1] < importances[f2]:
                    features_to_drop.add(f1)
                else:
                    features_to_drop.add(f2)

    final_features = [f for f in current_features if f not in features_to_drop]
    logging.info(f"Features dropped due to high correlation: {len(features_to_drop)}")
    logging.info(f"Number of final features: {len(final_features)}")
    return final_features

def check_leakage(X, y, feature_names, importance_df, threshold=0.5):
    logging.info("Starting leakage analysis...")
    leakage_report = []

    shap_importance = importance_df.set_index('Feature')['SHAP Importance (%)'] / 100
    max_importance = shap_importance.max()
    dominant_features = shap_importance[shap_importance > threshold].index.tolist()

    correlations = {}
    for feature in feature_names:
        if feature in X.columns:
            mask = ~X[feature].isna() & ~y.isna()
            if mask.sum() > 1:
                try:
                    corr, _ = spearmanr(X.loc[mask, feature], y[mask])
                    correlations[feature] = abs(corr)
                except Exception as e:
                    logging.warning(f"Error calculating Spearman correlation for leakage check on {feature}: {e}")

    for feature in feature_names:
        importance = shap_importance.get(feature, 0)
        correlation = correlations.get(feature, 0)
        is_dominant = feature in dominant_features
        is_high_corr = correlation > 0.9
        suspicion = is_dominant or is_high_corr
        leakage_report.append({
            'Feature': feature,
            'SHAP Importance (%)': importance * 100,
            'Spearman Correlation': correlation,
            'Suspected Leakage': suspicion,
            'Reason': 'Dominant Importance' if is_dominant else 'High Correlation' if is_high_corr else 'None'
        })

    leakage_df = pd.DataFrame(leakage_report)
    leakage_df.to_csv(LEAKAGE_ANALYSIS_FILE, index=False)
    logging.info(f"Leakage report saved to {LEAKAGE_ANALYSIS_FILE}")

    if leakage_df['Suspected Leakage'].any():
        logging.warning("Features suspected of leakage identified. Please review the report file.")
    else:
        logging.info("No signs of leakage found.")

    return leakage_df
    
# -------------------------------------------------------------------
# --- Main Execution ---
# -------------------------------------------------------------------

def main():
    clear_previous_files()
    start_time = time.time()
    logging.info(f"Starting feature selection script - version {CONFIG['version']}")

    # Step 1: Load Data
    logging.info("Step 1: Loading Data...")
    try:
        # Load training data
        X_train = pd.read_excel('Alldata_train.xlsx', engine='openpyxl')
        y_train = pd.read_excel('lreturns30w_train_scaled.xlsx', engine='openpyxl').iloc[:, 0].copy()
        y_train.name = 'lreturns30w'
        logging.info(f"Training features loaded: {X_train.shape}")
        logging.info(f"Training target loaded: {y_train.shape}")

        # Load testing data
        X_test = pd.read_excel('Alldata_test.xlsx', engine='openpyxl')
        y_test = pd.read_excel('lreturns30w_test_scaled.xlsx', engine='openpyxl').iloc[:, 0].copy()
        y_test.name = 'lreturns30w'
        logging.info(f"Testing features loaded: {X_test.shape}")
        logging.info(f"Testing target loaded: {y_test.shape}")

        # Drop 'Date' column if it exists
        if 'Date' in X_train.columns:
            X_train = X_train.drop(columns=['Date'])
            logging.info("Dropped 'Date' column from training data.")
        if 'Date' in X_test.columns:
            X_test = X_test.drop(columns=['Date'])
            logging.info("Dropped 'Date' column from testing data.")

        # Ensure no 'roc30' is accidentally dropped here as per the user's note
        #if 'roc30' in X_train.columns:
            #logging.info("'roc30' column present in training features.")
        #if 'roc30' in X_test.columns:
            #logging.info("'roc30' column present in testing features.")

        # Drop constant and near-constant features from training data
        cols_to_drop_train = [col for col in X_train.columns if X_train[col].nunique(dropna=False) <= 1]
        X_train = X_train.drop(columns=cols_to_drop_train) if cols_to_drop_train else X_train.copy()
        logging.info(f"Dropped {len(cols_to_drop_train)} constant/near-constant features from training data.")

        # Drop constant and near-constant features from testing data (using the same columns as training)
        X_test = X_test.drop(columns=[col for col in cols_to_drop_train if col in X_test.columns])
        logging.info(f"Dropped constant/near-constant features from testing data based on training data analysis.")

        # Remove target variable if present in features
        if y_train.name in X_train.columns:
            X_train = X_train.drop(columns=[y_train.name])
        if y_test.name in X_test.columns:
            X_test = X_test.drop(columns=[y_test.name])

        if not check_invalid_values(X_train, "Training Features") or not check_invalid_values(y_train.to_frame(), "Training Target"):
            raise ValueError("Training data contains invalid values.")
        if not check_invalid_values(X_test, "Testing Features") or not check_invalid_values(y_test.to_frame(), "Testing Target"):
            raise ValueError("Testing data contains invalid values.")

        # Handle missing values in training data by filling with the mean
        X_train = X_train.fillna(X_train.mean(numeric_only=True))
        if X_train.isna().any().any():
            X_train = X_train.dropna(axis=1, how='any')
            logging.warning("Dropped columns with remaining NaN in training data after mean imputation.")

        # Handle missing values in testing data using the mean from the training data
        for col in X_train.columns:
            if col in X_test.columns:
                X_test[col] = X_test[col].fillna(X_train[col].mean())
        # Drop any remaining NaN columns in test that were not in train
        X_test = X_test.dropna(axis=1, how='any')


        # Align columns between training and testing sets after feature engineering
        common_cols = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        logging.info(f"Aligned columns between training and testing sets. Final training features shape: {X_train.shape}, testing features shape: {X_test.shape}")

        # Ensure target indices are aligned with features
        X_train = X_train.loc[y_train.index]
        X_test = X_test.loc[y_test.index]

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        exit(1)

    # Step 2: Initial Filtering with Spearman (on training data only)
    logging.info("Step 2: Initial Filtering with Spearman (Training Data)...")
    spearman_selected_features = spearman_feature_selection(X_train, y_train, CONFIG['spearman_percentile'])
    if not spearman_selected_features:
        logging.error("No features selected by Spearman.")
        exit(1)
    with open(SPEARMAN_FILE, 'wb') as f:
        pickle.dump(spearman_selected_features, f)
    logging.info(f"Spearman selected features (training) saved.")

    X_train_spearman = X_train[spearman_selected_features]
    X_test_spearman = X_test[spearman_selected_features] # Apply same selection to test set
    logging.info(f"Number of features after Spearman: {len(spearman_selected_features)}")
    plot_correlation_heatmap(X_train_spearman, 'Correlation Heatmap of Spearman Features (Training)',
                             CORRELATION_HEATMAP_SPEARMAN_FILE)

    # Step 3: Bayesian Optimization (on training data only)
    logging.info("Step 3: Bayesian Optimization (Training Data)...")
    if not check_invalid_values(X_train_spearman, "X_train_spearman"):
        logging.error("Input data to BayesSearchCV is invalid.")
        exit(1)
    param_space = {
        'n_estimators': Integer(50, 500),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0)
    }
    xgb_base = XGBRegressor(random_state=42, objective='reg:squarederror')
    bayes_search = BayesSearchCV(
        estimator=xgb_base,
        search_spaces=param_space,
        n_iter=CONFIG['bayes_n_iter'],
        cv=KFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42),
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )
    with tqdm(total=CONFIG['bayes_n_iter'], desc="BayesSearchCV") as pbar:
        def callback(res):
            pbar.update(1)
        try:
            bayes_search.fit(X_train_spearman, y_train, callback=[callback, DeltaYStopper(
                delta=CONFIG['early_stopping_delta'], n_best=CONFIG['early_stopping_patience'])])
            best_params = bayes_search.best_params_
            with open(BAYES_OPT_FILE, 'wb') as f:
                pickle.dump({'best_params': best_params, 'best_score': bayes_search.best_score_}, f)
            logging.info("Bayesian optimization parameters (training) saved.")
        except Exception as e:
            logging.error(f"Error in BayesSearchCV: {e}")
            best_params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1,
                           'subsample': 0.8, 'colsample_bytree': 0.8}
            logging.warning(f"Using default parameters: {best_params}")

    best_params = {k: int(v) if isinstance(v, (np.integer, float)) and k in ['n_estimators', 'max_depth'] else float(v) if isinstance(v, (np.floating, float)) else v for k, v in best_params.items()}
    logging.info(f"Final parameters: {best_params}")

    # Step 4: Secondary Filtering (on training data only)
    logging.info("Step 4: Secondary Filtering (Training Data)...")
    final_selected_features = xgboost_correlation_filter(
        X_train_spearman, y_train, spearman_selected_features, best_params, CONFIG['correlation_threshold'])
    if not final_selected_features:
        logging.error("No features selected in secondary filtering. Stopping process.")
        exit(1)

    with open(XGB_FILTERED_FEATURES_FILE, 'wb') as f:
        pickle.dump(final_selected_features, f)
    pd.DataFrame({'Feature': final_selected_features}).to_csv(XGB_FILTERED_FEATURES_CSV_FILE, index=False)
    logging.info("Filtered features (training) saved.")

    X_train_final = X_train_spearman[final_selected_features]
    X_test_final = X_test_spearman[final_selected_features] # Apply same selection to test set
    logging.info(f"Number of final features: {len(final_selected_features)}")

    # Save training data with final features
    data_storage = {
        'X_train_final': X_train_final,
        'y_train': y_train,
        'final_selected_features': final_selected_features
    }
    with open(DATA_STORAGE_FILE, 'wb') as f:
        pickle.dump(data_storage, f)
    logging.info(f"Final training data saved to {DATA_STORAGE_FILE}.")

    # Step 5: Model Training and Evaluation (Training for CV, Test for final evaluation)
    logging.info("Step 5: Model Training and Evaluation...")
    final_model = XGBRegressor(random_state=42, **best_params)

    # Check if training data is not empty
    if X_train_final.empty or y_train.empty:
        logging.error("Training data is empty. Cannot train the model.")
        return

    kfold = KFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
    cv_results = {'Fold': range(1, CONFIG['k_folds'] + 1)}
    for metric, scorer in [('RMSE', 'neg_root_mean_squared_error'), ('MAE', 'neg_mean_absolute_error'), ('R2', 'r2')]:
        try:
            scores = cross_val_score(final_model, X_train_final, y_train, scoring=scorer, cv=kfold, n_jobs=-1)
            cv_results[metric] = -scores if 'neg_' in scorer else scores
            logging.info(f"CV {metric}: {cv_results[metric].mean():.6f} (±{cv_results[metric].std():.6f})")
        except Exception as e:
            logging.error(f"Error in CV {metric}: {e}")

    try:
        pd.DataFrame(cv_results).to_csv(CV_RESULTS_FILE, index=False)
        logging.info(f"CV results (training) saved to {CV_RESULTS_FILE}.")
    except Exception as e:
        logging.error(f"Error saving CV results: {e}")

    try:
        final_model.fit(X_train_final, y_train)
        logging.info("Final model trained on training data.")

        # Check if test features are not empty before prediction
        if X_test_final.empty:
            logging.warning("Test features are empty. Skipping prediction.")
            y_pred = None
        else:
            y_pred = final_model.predict(X_test_final)

        with open(get_path('final_model_train'), 'wb') as f:
            pickle.dump(final_model, f)
        logging.info(f"Final model (trained on training data) saved to {get_path('final_model_train')}.pkl.")
    except Exception as e:
        logging.error(f"Error in training or prediction: {e}")
        y_pred = None # Ensure y_pred is None if prediction fails

    if y_pred is not None:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mdape = median_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Test Results: RMSE={rmse:.6f}, MAE={mae:.6f}, MAPE={mape:.2f}%, MdAPE={mdape:.2f}%, R2={r2:.6f}")

        with open(TEST_EVALUATION_FILE, 'w') as f:
            f.write(f"Test Results:\n")
            f.write(f"RMSE={rmse:.6f}\n")
            f.write(f"MAE={mae:.6f}\n")
            f.write(f"MAPE={mape:.2f}%\n")
            f.write(f"MdAPE={mdape:.2f}%\n")
            f.write(f"R2={r2:.6f}\n")
        logging.info(f"Test evaluation metrics saved to {TEST_EVALUATION_FILE}")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs Actual')
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title(f'Actual vs Predicted (Test Set, R² = {r2:.3f})')
        plt.legend(); plt.grid(True); plt.savefig(ACTUAL_VS_PREDICTED_PLOT_FILE, dpi=300); plt.close()
        logging.info(f"Actual vs Predicted plot (test set) saved.")
    else:
        logging.warning("Skipping test evaluation and plotting due to prediction failure.")

    # Step 6: SHAP Analysis (on training data and model)
    logging.info("Step 6: SHAP Analysis (Training Data and Model)...")
    try:
        feature_importance = pd.Series(final_model.feature_importances_, index=X_train_final.columns).sort_values(ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_importance.head(50), y=feature_importance.head(50).index,
                    hue=feature_importance.head(50).index, palette='viridis', legend=False)
        plt.title('XGBoost Feature Importance (Trained on Training Data)')
        plt.savefig(XGBOOST_IMPORTANCE_PLOT_FILE, dpi=300); plt.close()

        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_train_final.sample(CONFIG['shap_sample_size'], random_state=42))
        shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X_train_final.columns).sort_values(ascending=False)
        shap_percentage = (shap_importance / shap_importance.sum()) * 100
        shap_df = shap_percentage.reset_index(name='SHAP Importance (%)').rename(columns={'index': 'Feature'})
        shap_df.to_csv(SHAP_IMPORTANCE_FILE, index=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='SHAP Importance (%)', y='Feature', data=shap_df.head(50),
                    hue='Feature', palette='viridis', legend=False)
        plt.title('SHAP Feature Importance (Trained on Training Data)')
        plt.savefig(SHAP_IMPORTANCE_PLOT_FILE, dpi=300); plt.close()

        shap.summary_plot(shap_values, X_train_final.sample(CONFIG['shap_sample_size'], random_state=42), show=False)
        plt.savefig(SHAP_SUMMARY_PLOT_FILE, dpi=300); plt.close()
        logging.info("SHAP analysis completed on training data and model.")
    except Exception as e:
        logging.error(f"Error in SHAP: {e}")

    # Step 7: Leakage Analysis (on training data)
    logging.info("Step 7: Leakage Analysis (Training Data)...")
    try:
        leakage_df = check_leakage(X_train_final, y_train, final_selected_features, shap_df,
                                     CONFIG['leakage_importance_threshold'])
    except Exception as e:
        logging.error(f"Error in leakage analysis: {e}")


# Step 8: Final Saving
    logging.info("Step 8: Final Saving...")
    pd.DataFrame({'Feature': final_selected_features}).to_csv(FINAL_FEATURES_LIST_FILE, index=False)

    # Save final selected features for training data
    final_features_train_filename = 'important_features_train_v8_separate_train_test.csv'
    try:
        final_data_train = X_train_final[final_selected_features]
        final_data_train.to_csv(final_features_train_filename, index=False)
        logging.info(f"Final selected features and data for training saved to {final_features_train_filename}.")
        plot_correlation_heatmap(final_data_train, 'Correlation Heatmap of Final Features (Training)', CORRELATION_HEATMAP_FINAL_FILE)
    except Exception as e:
        logging.error(f"Error saving final training data with selected features: {e}")

    # Save final selected features and data for testing data
    final_features_test_filename = 'important_features_test_v8_separate_train_test.csv'
    try:
        final_data_test = X_test_final[final_selected_features]
        final_data_test.to_csv(final_features_test_filename, index=False)
        logging.info(f"Final selected features and data for testing saved to {final_features_test_filename}.")
        # We don't plot a correlation heatmap for the test features here as the selection was based on training data.
    except Exception as e:
        logging.error(f"Error saving final testing data with selected features: {e}")

    # Save training data with final features (this part remains as before, saving the actual data to a different file)
    try:
        final_data_train_full = X_train_final[final_selected_features]
        final_data_train_full.to_csv(FINAL_FEATURES_FILE, index=False)
        logging.info(f"Final training data with selected features saved to {FINAL_FEATURES_FILE}.")
        # plot_correlation_heatmap is already called above for this
    except Exception as e:
        logging.error(f"Error saving final training data: {e}")

if __name__ == "__main__":
    main()
