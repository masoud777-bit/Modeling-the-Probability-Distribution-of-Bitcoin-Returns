"""
Author: [M.fadakar]
"""

import warnings
import logging
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import GradientBoostingRegressor
import properscoring as ps
from scipy.stats import norm, entropy, wasserstein_distance
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.svm import SVR

# Initial settings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SEED = 42
np.random.seed(SEED)

# Data paths
TRAIN_FEATURES = r"C:\Users\masoud\important_features_train_v8_separate_train_test.xlsx"
TRAIN_TARGET = r"C:\Users\masoud\lreturns30w_train_scaled.xlsx"
TEST_FEATURES = r"C:\Users\masoud\important_features_test_v8_separate_train_test.xlsx"
TEST_TARGET = r"C:\Users\masoud\lreturns30w_test_scaled.xlsx"
SCALER_PATH = r"C:\Users\masoud\robust_scaler.pkl"
SAVE_DIR = r"C:\Users\masoud\results\P\baseline_comparison"

# Function to load data
def load_data(features_path, target_path):
    try:
        X = pd.read_excel(features_path).values.astype(np.float32)
        y = pd.read_excel(target_path)['lreturn'].values.astype(np.float32).flatten()
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Data mismatch: X has {X.shape[0]} samples and y has {y.shape[0]} samples")
        logging.info(f"Data loaded: X shape {X.shape}, y shape {y.shape}")
        return X, y
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

# Function to generate quantile predictions
def generate_quantile_predictions(mean_preds, errors_std, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    quantile_preds = {}
    if not isinstance(errors_std, (int, float, np.ndarray)):
        logging.warning("Invalid errors_std type in generate_quantile_predictions. Using 0.1.")
        errors_std = 0.1
    if isinstance(errors_std, np.ndarray) and errors_std.shape != mean_preds.shape and errors_std.size != 1:
        logging.warning(f"Shape of errors_std ({errors_std.shape}) does not match mean_preds ({mean_preds.shape}) in generate_quantile_predictions. Using mean of errors_std.")
        errors_std = np.mean(errors_std) if errors_std.size > 0 else 0.1
    for q_val in quantiles:
        quantile_preds[q_val] = mean_preds + norm.ppf(q_val) * errors_std
    return quantile_preds

# Function to calculate PIT
def calculate_pit(y_true, quantile_preds, quantiles):
    if quantile_preds is None or not quantile_preds:
        return np.nan, np.nan
    pit_values = []
    valid_quantiles_in_dict = sorted([q for q in quantiles if q in quantile_preds and quantile_preds[q] is not None])
    if not valid_quantiles_in_dict or len(valid_quantiles_in_dict) < 2:
        logging.warning("Insufficient valid quantiles to calculate PIT.")
        return np.nan, np.nan
    for i, y_val in enumerate(y_true):
        pred_values_for_point = []
        current_qs_for_point = []
        for q_k in valid_quantiles_in_dict:
            if i < len(quantile_preds[q_k]):
                pred_values_for_point.append(quantile_preds[q_k][i])
                current_qs_for_point.append(q_k)
            else:
                logging.warning(f"Length of prediction for quantile {q_k} does not match length of y_true at point {i}.")
        if not pred_values_for_point or len(pred_values_for_point) < 2:
            pit_values.append(np.nan)
            continue
        y_val_pit = np.nan
        if y_val <= pred_values_for_point[0]:
            y_val_pit = current_qs_for_point[0]
        elif y_val >= pred_values_for_point[-1]:
            y_val_pit = current_qs_for_point[-1]
        else:
            for j in range(len(pred_values_for_point) - 1):
                if pred_values_for_point[j] <= y_val <= pred_values_for_point[j+1]:
                    q_lower = current_qs_for_point[j]
                    q_upper = current_qs_for_point[j+1]
                    val_lower = pred_values_for_point[j]
                    val_upper = pred_values_for_point[j+1]
                    if val_upper == val_lower:
                        y_val_pit = q_upper
                    else:
                        weight = (y_val - val_lower) / (val_upper - val_lower)
                        y_val_pit = q_lower + weight * (q_upper - q_lower)
                    break
        pit_values.append(y_val_pit)
    pit_values_arr = np.array(pit_values)
    if np.all(np.isnan(pit_values_arr)):
        return np.nan, np.nan
    return np.nanmean(pit_values_arr), np.nanstd(pit_values_arr)

# Function to calculate quantile loss
def calculate_quantile_loss(y_true, pred_quantile, q):
    errors = y_true - pred_quantile
    return np.mean(np.maximum(q * errors, (q - 1) * errors))

# Function to calculate evaluation metrics
def calculate_metrics(y_true, y_pred, y_pred_lower, y_pred_upper, naive_error, scaler=None, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], predictions_quantiles=None, model_name="Model"):
    metrics = {}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    interval_width = np.abs(y_pred_upper - y_pred_lower)
    msis_corrected = np.mean(interval_width) / naive_error if naive_error != 0 and not np.isnan(naive_error) else np.inf
    if predictions_quantiles is None or not predictions_quantiles:
        logging.warning(f"predictions_quantiles for {model_name} not provided or empty. Generating from y_pred and overall error.")
        errors_std_fallback = np.std(y_true - y_pred)
        if errors_std_fallback < 1e-9: errors_std_fallback = 0.1
        predictions_quantiles = generate_quantile_predictions(y_pred, errors_std_fallback, quantiles)
    crps_calculated = np.nan
    try:
        sorted_q_keys = sorted([q for q in predictions_quantiles.keys() if predictions_quantiles[q] is not None and len(predictions_quantiles[q]) == len(y_true)])
        if len(sorted_q_keys) > 1:
            ensemble_preds_list = [predictions_quantiles[q] for q in sorted_q_keys]
            ensemble_preds = np.column_stack(ensemble_preds_list)
            crps_calculated = ps.crps_ensemble(y_true, ensemble_preds).mean()
        else:
            crps_calculated = np.mean(np.abs(y_true - y_pred))
    except Exception as e:
        logging.error(f"Error calculating CRPS for {model_name}: {e}")
        crps_calculated = np.mean(np.abs(y_true - y_pred))
    pit_mean, pit_sd = calculate_pit(y_true, predictions_quantiles, quantiles)
    ql_values = {}
    default_ql_value_for_metric = 0.5
    for q_target in quantiles:
        ql_key_name = f'ql{int(q_target*100)}'
        if q_target in predictions_quantiles and predictions_quantiles[q_target] is not None:
            ql_values[ql_key_name] = calculate_quantile_loss(y_true, predictions_quantiles[q_target], q_target)
        elif q_target == default_ql_value_for_metric:
            ql_values[ql_key_name] = calculate_quantile_loss(y_true, y_pred, q_target)
        else:
            logging.debug(f"Quantile {q_target} for quantile loss calculation not found or None in {model_name}.")
            ql_values[ql_key_name] = np.nan
    qlm = np.nanmean([ql_values[key] for key in ql_values if not np.isnan(ql_values[key])])
    kl_div = np.nan
    wasserstein = np.nan
    try:
        if len(y_true) > 1 and len(y_pred) == len(y_true) and np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
            hist_true, bins = np.histogram(y_true, bins=50, density=True)
            hist_pred, _ = np.histogram(y_pred, bins=bins, density=True)
            hist_true = np.maximum(hist_true, 1e-10)
            hist_pred = np.maximum(hist_pred, 1e-10)
            kl_div = entropy(hist_true, hist_pred)
            wasserstein = wasserstein_distance(y_true, y_pred)
        else:
            logging.warning(f"Insufficient data or variance to calculate KL/Wasserstein for {model_name} (y_true_std: {np.std(y_true):.2e}, y_pred_std: {np.std(y_pred):.2e}).")
    except Exception as e:
        logging.warning(f"Calculation of KL divergence or Wasserstein distance for {model_name} failed: {str(e)}")
    metrics.update({
        'crps': crps_calculated, 'kl_divergence': kl_div, 'wasserstein': wasserstein,
        'msis': msis_corrected, 'pit_mean': pit_mean, 'pit_sd': pit_sd,
        'rmse': rmse, 'mae': mae, 'r2': r2, 'qlm': qlm
    })
    metrics.update(ql_values)
    if scaler:
        try:
            y_true_orig = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            metrics['rmse_original'] = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
            metrics['mae_original'] = mean_absolute_error(y_true_orig, y_pred_orig)
            metrics['r2_original'] = r2_score(y_true_orig, y_pred_orig)
            # Calculate CRPS in the original scale
            ensemble_preds_orig = np.column_stack([scaler.inverse_transform(predictions_quantiles[q].reshape(-1, 1)).flatten() for q in sorted_q_keys])
            crps_original = ps.crps_ensemble(y_true_orig, ensemble_preds_orig).mean()
            metrics['crps_original'] = crps_original
        except Exception as e:
            logging.warning(f"Calculation of metrics in original scale for {model_name} failed: {str(e)}. (Possibly due to scaler loading error)")
    return metrics

# Function for quantile regression forecast
def quantile_regression_forecast(X, y, X_test, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], model_name_log="QuantileRegression"):
    predictions = {}
    if X is None or y is None or X_test is None: raise ValueError("Input data cannot be empty")
    X, y, X_test = np.asarray(X), np.asarray(y).reshape(-1), np.asarray(X_test)
    if X.shape[0] != y.shape[0]: raise ValueError(f"Dimension mismatch between X and y")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)) or np.any(np.isnan(X_test)): raise ValueError("Input data contains NaN values")
    for q in quantiles:
        try:
            model = QuantileRegressor(quantile=q, alpha=0, solver='revised simplex')
            model.fit(X, y)
            predictions[q] = model.predict(X_test)
        except Exception as e_qr_revised_simplex:
            logging.warning(f"QuantileRegressor with solver='revised simplex' for quantile {q} failed: {str(e_qr_revised_simplex)}. Using fallback mechanism for this quantile.")
            y_mean_fallback = np.mean(y)
            y_std_fallback = np.std(y)
            if y_std_fallback < 1e-9: y_std_fallback = 0.1
            predictions[q] = y_mean_fallback + norm.ppf(q) * y_std_fallback * np.ones(X_test.shape[0])
    y_pred_fallback_mean = np.mean(y) * np.ones(X_test.shape[0])
    y_pred_fallback_std = np.std(y)
    if y_pred_fallback_std < 1e-9: y_pred_fallback_std = 0.1
    final_pred_05 = predictions.get(0.5, y_pred_fallback_mean)
    final_pred_01 = predictions.get(0.1, y_pred_fallback_mean + norm.ppf(0.1) * y_pred_fallback_std)
    final_pred_09 = predictions.get(0.9, y_pred_fallback_mean + norm.ppf(0.9) * y_pred_fallback_std)
    for q_ensure in quantiles:
        if q_ensure not in predictions or predictions[q_ensure] is None:
            predictions[q_ensure] = y_pred_fallback_mean + norm.ppf(q_ensure) * y_pred_fallback_std
        if q_ensure == 0.5: final_pred_05 = predictions[q_ensure]
        if q_ensure == 0.1: final_pred_01 = predictions[q_ensure]
        if q_ensure == 0.9: final_pred_09 = predictions[q_ensure]
    return final_pred_05, final_pred_01, final_pred_09, predictions

# Function for random forest quantile forecast
def random_forest_quantile_forecast(X, y, X_test, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], model_name_log="RandomForest"):
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, max_depth=None)
        model.fit(X, y)
        all_tree_preds = np.array([tree.predict(X_test) for tree in model.estimators_])
        predictions = {q: np.quantile(all_tree_preds, q, axis=0) for q in quantiles}
        return predictions[0.5], predictions[0.1], predictions[0.9], predictions
    except Exception as e:
        logging.error(f"Error in {model_name_log}: {str(e)}. Using fallback mechanism.")
        try:
            model_fallback = RandomForestRegressor(n_estimators=50, random_state=SEED, n_jobs=-1, max_depth=10)
            model_fallback.fit(X, y)
            y_pred_fallback = model_fallback.predict(X_test)
            train_preds_fallback = model_fallback.predict(X)
            errors_std_fallback = np.std(y - train_preds_fallback) if len(y - train_preds_fallback) > 0 else 0.1
            if errors_std_fallback < 1e-9: errors_std_fallback = 0.1
            predictions_fallback = generate_quantile_predictions(y_pred_fallback, errors_std_fallback, quantiles)
            return predictions_fallback[0.5], predictions_fallback[0.1], predictions_fallback[0.9], predictions_fallback
        except Exception as e_fallback_rf:
            logging.error(f"Error in fallback {model_name_log}: {str(e_fallback_rf)}.")
            y_mean = np.mean(y); y_std = np.std(y)
            if y_std < 1e-9: y_std = 0.1
            y_pred_final_fallback = np.full(X_test.shape[0], y_mean)
            predictions_final_fallback = generate_quantile_predictions(y_pred_final_fallback, y_std, quantiles)
            return predictions_final_fallback[0.5], predictions_final_fallback[0.1], predictions_final_fallback[0.9], predictions_final_fallback

# Function for gradient boosting forecast
def gradient_boosting_forecast(X, y, X_test, model_type='LightGBM', quantiles_to_predict=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    model_name_log = model_type
    quantile_preds = {}
    try:
        common_lgbm_quantile_params = {
            'objective': 'quantile', 'metric': 'quantile', 'random_state': SEED,
            'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 31,
            'min_child_samples': 20, 'min_sum_hessian_in_leaf': 1e-3,
            'verbose': -1, 'n_jobs': -1
        }
        if model_type == 'LightGBM':
            for q in quantiles_to_predict:
                lgbm_params = {**common_lgbm_quantile_params, 'alpha': q}
                model = lgb.LGBMRegressor(**lgbm_params)
                model.fit(X, y)
                quantile_preds[q] = model.predict(X_test)
        elif model_type == 'CatBoost':
            for q in quantiles_to_predict:
                cb_params = {'loss_function': f'Quantile:alpha={q}', 'random_seed': SEED, 'verbose': 0, 'n_estimators': 100, 'learning_rate': 0.05}
                model = cb.CatBoostRegressor(**cb_params)
                model.fit(X, y)
                quantile_preds[q] = model.predict(X_test)
        elif model_type == 'GradientBoosting(sklearn)':
            model_name_log = "GradientBoosting(sklearn)"
            for q in quantiles_to_predict:
                gbr_params = {'loss': 'quantile', 'alpha': q, 'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3, 'random_state': SEED}
                model = GradientBoostingRegressor(**gbr_params)
                model.fit(X,y)
                quantile_preds[q] = model.predict(X_test)
        elif model_type == 'AdaBoost':
            base_estimator = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=SEED)
            model = AdaBoostRegressor(estimator=base_estimator, n_estimators=50, random_state=SEED)
            model.fit(X, y)
            y_pred_adaboost_mean = model.predict(X_test)
            train_preds_adaboost = model.predict(X)
            errors_std_adaboost = np.std(y - train_preds_adaboost) if len(y - train_preds_adaboost) > 0 else 0.1
            if errors_std_adaboost < 1e-9: errors_std_adaboost = 0.1
            quantile_preds = generate_quantile_predictions(y_pred_adaboost_mean, errors_std_adaboost, quantiles_to_predict)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        y_mean_fb = np.mean(y); y_std_fb = np.std(y);
        if y_std_fb < 1e-9: y_std_fb = 0.1
        pred_05 = quantile_preds.get(0.5, y_mean_fb + norm.ppf(0.5) * y_std_fb * np.ones(X_test.shape[0]))
        pred_01 = quantile_preds.get(0.1, y_mean_fb + norm.ppf(0.1) * y_std_fb * np.ones(X_test.shape[0]))
        pred_09 = quantile_preds.get(0.9, y_mean_fb + norm.ppf(0.9) * y_std_fb * np.ones(X_test.shape[0]))
        for q_fill in [0.1, 0.5, 0.9]:
            if q_fill not in quantile_preds or quantile_preds[q_fill] is None:
                quantile_preds[q_fill] = y_mean_fb + norm.ppf(q_fill) * y_std_fb * np.ones(X_test.shape[0])
        return pred_05, pred_01, pred_09, quantile_preds
    except Exception as e:
        logging.error(f"Error in {model_name_log}: {str(e)}. Using general fallback mechanism.")
        y_mean_fallback = np.mean(y); y_std_fallback = np.std(y)
        if y_std_fallback < 1e-9: y_std_fallback = 0.1
        y_pred_final_fallback = np.full(X_test.shape[0], y_mean_fallback)
        predictions_final_fallback = generate_quantile_predictions(y_pred_final_fallback, y_std_fallback, quantiles_to_predict)
        return predictions_final_fallback[0.5], predictions_final_fallback[0.1], predictions_final_fallback[0.9], predictions_final_fallback

# Function for SVR forecast
def svr_forecast(X, y, X_test, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], model_name_log="SVR"):
    try:
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        model.fit(X, y)
        y_pred = model.predict(X_test)
        errors_std = np.std(y - model.predict(X))
        if errors_std < 1e-9: errors_std = 0.1
        predictions_quantiles = generate_quantile_predictions(y_pred, errors_std, quantiles)
        return y_pred, predictions_quantiles[0.1], predictions_quantiles[0.9], predictions_quantiles
    except Exception as e:
        logging.error(f"Error in {model_name_log}: {str(e)}. Using fallback mechanism.")
        y_mean = np.mean(y); y_std = np.std(y)
        if y_std < 1e-9: y_std = 0.1
        y_pred_final_fallback = np.full(X_test.shape[0], y_mean)
        predictions_final_fallback = generate_quantile_predictions(y_pred_final_fallback, y_std, quantiles)
        return predictions_final_fallback[0.5], predictions_final_fallback[0.1], predictions_final_fallback[0.9], predictions_final_fallback

# Main function
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_run_folder_name = f"run_{timestamp}"
    result_dir = os.path.join(SAVE_DIR, unique_run_folder_name)
    os.makedirs(result_dir, exist_ok=True)
    try:
        X_train, y_train = load_data(TRAIN_FEATURES, TRAIN_TARGET)
        X_test, y_test = load_data(TEST_FEATURES, TEST_TARGET)
    except Exception as e:
        logging.error(f"Program stopped due to error in data loading: {e}")
        return
    naive_error = np.nan
    if len(y_train) > 1:
        naive_error = mean_absolute_error(y_train[1:], y_train[:-1])
    if np.isnan(naive_error) or naive_error < 1e-9:
        logging.warning(f"Naive error (naive_error) not calculated or very small ({naive_error}). MSIS may be meaningless. Using default value 0.1.")
        naive_error = 0.1
    scaler = None
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        logging.info("Scaler loaded successfully")
    except ModuleNotFoundError as e_scaler_module:
        logging.warning(f"Error loading scaler (module not found): {e_scaler_module}. Original scale metrics will not be calculated.")
        scaler = None
    except FileNotFoundError:
        logging.warning(f"Scaler file not found at {SCALER_PATH}. Original scale metrics will not be calculated.")
        scaler = None
    except Exception as e_scaler_other:
        logging.warning(f"Other error loading scaler: {e_scaler_other}. Original scale metrics will not be calculated.")
        scaler = None
    models = {
        'QuantileReg': lambda: quantile_regression_forecast(X_train, y_train, X_test, model_name_log='QuantileReg'),
        'RandomForestQ': lambda: random_forest_quantile_forecast(X_train, y_train, X_test, model_name_log='RandomForestQ'),
        'LightGBMQ': lambda: gradient_boosting_forecast(X_train, y_train, X_test, model_type='LightGBM'),
        'CatBoostQ': lambda: gradient_boosting_forecast(X_train, y_train, X_test, model_type='CatBoost'),
        'GBoost(skl)Q': lambda: gradient_boosting_forecast(X_train, y_train, X_test, model_type='GradientBoosting(sklearn)'),
        'AdaBoostQ': lambda: gradient_boosting_forecast(X_train, y_train, X_test, model_type='AdaBoost'),
        'SVR': lambda: svr_forecast(X_train, y_train, X_test, model_name_log='SVR')
    }
    results = {}
    all_predictions_df_data = {'actual': y_test.copy()}
    quantiles_for_metrics_and_plots = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for name, func in models.items():
        logging.info(f"Running {name}...")
        try:
            y_pred, y_lower, y_upper, predictions_quantiles_dict = func()
            if y_pred is None:
                logging.error(f"{name} returned None for y_pred. Skipping.")
                nan_metrics = {k: np.nan for k in ['crps', 'kl_divergence', 'wasserstein', 'msis', 'pit_mean', 'pit_sd', 'rmse', 'mae', 'r2', 'qlm', 'crps_original'] + [f'ql{int(q*100)}' for q in quantiles_for_metrics_and_plots]}
                results[name] = nan_metrics
                continue
            target_len = len(y_test)
            def adjust_len(arr, name_arr):
                if arr is None: return None
                if len(arr) == target_len: return arr
                logging.warning(f"Length of {name_arr} in {name} ({len(arr)}) differs from ({target_len}). Adjusting.")
                if len(arr) > target_len: return arr[:target_len]
                else:
                    pad_val = np.mean(arr) if len(arr) > 0 else np.mean(y_test)
                    return np.pad(arr, (0, target_len - len(arr)), 'constant', constant_values=pad_val)
            y_pred = adjust_len(y_pred, "y_pred")
            y_lower = adjust_len(y_lower, "y_lower")
            y_upper = adjust_len(y_upper, "y_upper")
            if predictions_quantiles_dict:
                for q_key, pq_arr_val in predictions_quantiles_dict.items():
                    predictions_quantiles_dict[q_key] = adjust_len(pq_arr_val, f"quantile_{q_key}")
            else:
                predictions_quantiles_dict = {}
            current_metrics = calculate_metrics(
                y_test, y_pred, y_lower, y_upper,
                naive_error, scaler,
                quantiles=quantiles_for_metrics_and_plots,
                predictions_quantiles=predictions_quantiles_dict,
                model_name=name
            )
            results[name] = current_metrics
            all_predictions_df_data[f'{name}_median'] = y_pred
            all_predictions_df_data[f'{name}_lower01'] = y_lower
            all_predictions_df_data[f'{name}_upper09'] = y_upper
            if predictions_quantiles_dict:
                for q_plot, pred_q_plot_val in predictions_quantiles_dict.items():
                    if pred_q_plot_val is not None:
                        all_predictions_df_data[f'{name}_q{int(q_plot*100)}'] = pred_q_plot_val
            plt.figure(figsize=(12, 6), facecolor='white')
            plt.plot(y_test, label='Actual', color='blue', linewidth=1.5, alpha=0.7)
            if y_pred is not None: plt.plot(y_pred, label=f'{name} Prediction (0.5)', color='orange', linewidth=1.5)
            if y_lower is not None and y_upper is not None:
                plt.fill_between(range(len(y_test)), y_lower, y_upper, color='orange', alpha=0.2, label='90% Interval')
            plt.title(f'{name} Predictions vs Actual Values'); plt.xlabel('Time Step'); plt.ylabel('Value')
            plt.legend(); plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
            plt.savefig(os.path.join(result_dir, f'{name}_predictions.png'), dpi=300, facecolor='white')
            plt.close()
            if predictions_quantiles_dict:
                plt.figure(figsize=(8, 8), facecolor='white')
                valid_q_keys_for_calib = sorted([q for q, arr in predictions_quantiles_dict.items() if arr is not None and len(arr) == len(y_test)])
                if len(valid_q_keys_for_calib) >=2:
                    empirical_probs_calib = [np.mean(y_test <= predictions_quantiles_dict[q_calib]) for q_calib in valid_q_keys_for_calib]
                    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
                    plt.plot(valid_q_keys_for_calib, empirical_probs_calib, 'o-', label=f'{name} Calibration', markersize=5)
                    plt.xlabel('Predicted Quantile'); plt.ylabel('Observed Frequency')
                    plt.title(f'Calibration Plot for {name}'); plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.5); plt.axis([0, 1, 0, 1]); plt.tight_layout()
                    plt.savefig(os.path.join(result_dir, f'{name}_calibration.png'), dpi=300, facecolor='white')
                    plt.close()
        except Exception as e_model_run:
            logging.error(f"General error in model {name} execution: {str(e_model_run)}")
            nan_metrics = {k: np.nan for k in ['crps', 'kl_divergence', 'wasserstein', 'msis', 'pit_mean', 'pit_sd', 'rmse', 'mae', 'r2', 'qlm', 'crps_original'] + [f'ql{int(q*100)}' for q in quantiles_for_metrics_and_plots]}
            results[name] = nan_metrics
            nan_array = np.full(len(y_test), np.nan)
            all_predictions_df_data[f'{name}_median'] = nan_array
            all_predictions_df_data[f'{name}_lower01'] = nan_array
            all_predictions_df_data[f'{name}_upper09'] = nan_array
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(result_dir, 'metrics_comparison.csv'))
    logging.info(f"Metrics saved to {result_dir}/metrics_comparison.csv")
    final_df_cols = {}
    ref_len = len(y_test)
    for col, arr_data in all_predictions_df_data.items():
        if arr_data is None:
            final_df_cols[col] = np.full(ref_len, np.nan)
        elif len(arr_data) != ref_len:
            logging.warning(f"Length of column '{col}' ({len(arr_data)}) in final DataFrame differs from ({ref_len}). Padding with NaN.")
            final_df_cols[col] = np.full(ref_len, np.nan)
        else:
            final_df_cols[col] = arr_data
    all_predictions_final_df = pd.DataFrame(final_df_cols)
    all_predictions_final_df.to_csv(os.path.join(result_dir, 'all_model_predictions.csv'), index=False)
    logging.info(f"All predictions saved to {result_dir}/all_model_predictions.csv")
    smaller_is_better = ['crps', 'kl_divergence', 'wasserstein', 'msis', 'pit_mean', 'pit_sd', 'rmse', 'qlm', 'ql50', 'ql90']
    larger_is_better = ['r2']
    metrics_to_plot_combined = [m for m in smaller_is_better if m in results_df.columns] + [m for m in larger_is_better if m in results_df.columns]
    num_metrics = len(metrics_to_plot_combined)
    if num_metrics == 0:
        logging.warning("No metrics found for comparison plot.")
        return
    cols_plot = min(3, num_metrics)
    rows_plot = int(np.ceil(num_metrics / cols_plot))
    plt.figure(figsize=(cols_plot * 6, rows_plot * 5), facecolor='white')
    for i, metric in enumerate(metrics_to_plot_combined):
        ax = plt.subplot(rows_plot, cols_plot, i + 1)
        valid_metric_values = {name: res.get(metric, np.nan) for name, res in results.items()}
        valid_metric_values = {k: v for k, v in valid_metric_values.items() if not np.isnan(v)}
        if valid_metric_values:
            sort_ascending = metric in smaller_is_better
            sorted_models_plot = sorted(valid_metric_values.keys(), key=lambda k: valid_metric_values[k], reverse=not sort_ascending)
            sorted_values_plot = [valid_metric_values[k] for k in sorted_models_plot]
            bars = ax.bar(range(len(sorted_models_plot)), sorted_values_plot, color='skyblue', width=0.6)
            ax.set_xticks(range(len(sorted_models_plot)))
            ax.set_xticklabels(sorted_models_plot, rotation=60, ha='right', fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_title(f'{metric.upper()}', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.4, axis='y')
            for bar_item in bars:
                yval = bar_item.get_height()
                va_text = 'bottom' if yval >= 0 else 'top'
                offset = 0.01 * max(np.abs(sorted_values_plot)) if sorted_values_plot else 0.01
                text_y_pos = yval + offset * np.sign(yval) if yval != 0 else offset
                ax.text(bar_item.get_x() + bar_item.get_width()/2.0, text_y_pos,
                        f'{yval:.3f}', ha='center', va=va_text, fontsize=7, color='black')
        else:
            ax.text(0.5, 0.5, "No data to display", ha='center', va='center', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{metric.upper()}', fontsize=10)
    plt.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(result_dir, 'model_performance_comparison.png'), dpi=300, facecolor='white')
    plt.close()
    logging.info(f"Analysis complete. Results saved to {result_dir}")

if __name__ == "__main__":
    main()
