# -*- coding: utf-8 -*-
"""
Spline Quantile LSTM for Probabilistic Time Series Forecasting.

This script implements a sophisticated machine learning model for time series
forecasting, leveraging Long Short-Term Memory (LSTM) networks combined with
spline-based quantile regression. The model is designed to predict a full
probability distribution of future values rather than a single point estimate,
making it particularly useful for risk assessment and decision-making under
uncertainty.

The key features of this implementation include:
- A custom LSTM architecture for learning temporal dependencies.
- A spline-based output layer to efficiently model a large number of quantiles.
- A quantile loss function (pinball loss) for training the model.
- Isotonic regression for post-hoc calibration of predicted distributions.
- Hyperparameter tuning using Ray Tune with an Asynchronous Successive Halving
  Algorithm (ASHA) and HyperOpt search.
- Comprehensive evaluation metrics, including CRPS, MSIS, and PIT histograms,
  to assess the quality of the probabilistic forecasts.

The script is structured to be modular and reproducible, with clear separation
of concerns for data loading, model building, training, calibration, and
evaluation.
Author: [M.fadakar]
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

import os
import random
import pickle
import warnings
import logging
from datetime import datetime
from typing import Tuple, Dict, Optional, List

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import properscoring as ps
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.isotonic import IsotonicRegression
from scipy.stats import wasserstein_distance

# Ray Tune for hyperparameter optimization
import ray
from ray import tune
from ray.train import report
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch


# =============================================================================
# 2. CONFIGURATION AND INITIALIZATION
# =============================================================================

# --- General Settings ---
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.WARNING)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Reproducibility ---
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# --- Ray Initialization ---
if not ray.is_initialized():
    ray.init(
        num_cpus=4,
        num_gpus=0,
        ignore_reinit_error=True,
        log_to_driver=False
    )

# --- Logging Setup ---
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(log_dir, f'spline_quantile_lstm_tune_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def custom_trial_dirname_creator(trial: tune.Trial) -> str:
    """
    Creates a custom directory name for a Ray Tune trial.

    Args:
        trial (tune.Trial): The Ray Tune trial object.

    Returns:
        str: A formatted string with the trial ID.
    """
    return f"trial_{trial.trial_id[:8]}"


# =============================================================================
# 4. MAIN CLASS: SplineQuantileRegressionModel
# =============================================================================

class SplineQuantileRegressionModel:
    def __init__(self, config: Dict):
        self.time_steps = int(float(config['time_steps']))
        self.num_knots = int(float(config.get('num_knots', 30)))
        self.knots = np.linspace(0, 1, self.num_knots, dtype=np.float32)

        low_quantiles = np.concatenate([np.linspace(i / 100, (i + 1) / 100, 17, dtype=np.float32)[:-1] for i in range(1, 10)])
        middle_quantiles = np.concatenate([np.linspace(i / 100, (i + 1) / 100, 13, dtype=np.float32)[:-1] for i in range(10, 90)])
        high_quantiles = np.concatenate([np.linspace(i / 100, (i + 1) / 100, 17, dtype=np.float32)[:-1] for i in range(90, 100)])
        self.quantiles = np.concatenate((low_quantiles, middle_quantiles, high_quantiles))

        M = np.zeros((len(self.quantiles), len(self.knots)), dtype=np.float32)
        for k, tau in enumerate(self.quantiles):
            j = np.searchsorted(self.knots, tau, side='right') - 1
            if j < 0:
                j = 0
            elif j >= len(self.knots) - 1:
                j = len(self.knots) - 2
            knot_j = self.knots[j]
            knot_j1 = self.knots[j + 1]
            w_k = (tau - knot_j) / (knot_j1 - knot_j)
            M[k, j] = 1 - w_k
            M[k, j + 1] = w_k
        self.M = tf.constant(M, dtype=tf.float32)

        self.model = None
        self.naive_MAE = None
        self.history = None
        self.config = config
    
        self.es_patience = int(float(config.get('es_patience', 10)))
        self.rlr_factor = float(config.get('rlr_factor', 0.2))
        self.rlr_patience = int(float(config.get('rlr_patience', 13)))
        self.rlr_min_lr = float(config.get('rlr_min_lr', 1e-7))
        self.rlr_cooldown = int(float(config.get('rlr_cooldown', 3)))
        self.ir_pit_to_rank = None
        self.ir_rank_to_pit = None
        logging.info("SplineQuantileRegressionModel initialized")

    def load_data(self, features_path: str, target_path: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if not os.path.exists(features_path) or not os.path.exists(target_path):
                raise FileNotFoundError(f"Data file not found: {features_path} or {target_path}")
            X = pd.read_excel(features_path).values.astype(np.float32)
            y_df = pd.read_excel(target_path)
            if 'lreturn' not in y_df.columns:
                raise ValueError(f"Column 'lreturn' not found in target file: {target_path}")
            y = y_df['lreturn'].values.astype(np.float32).flatten()
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Data mismatch: X has {X.shape[0]} samples, y has {y.shape[0]} samples")
            logging.info(f"Data loaded: X shape {X.shape}, y shape {y.shape}")
            return X, y
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def reshape_for_lstm(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        if n_samples < self.time_steps:
            raise ValueError(f"Number of samples ({n_samples}) is less than time_steps ({self.time_steps})")
        X_reshaped = []
        y_reshaped = []
        valid_indices = []
        for i in range(n_samples - self.time_steps + 1):
            X_window = X[i:i + self.time_steps, :]
            X_reshaped.append(X_window)
            y_reshaped.append(y[i + self.time_steps - 1])
            valid_indices.append(i + self.time_steps - 1)
        return np.array(X_reshaped), np.array(y_reshaped), np.array(valid_indices)

    def build_model(self, num_features: int) -> Model:
        inputs = Input(shape=(self.time_steps, num_features))
        x = inputs
        for _ in range(self.config['num_lstm_layers']):
            x = LSTM(
                units=self.config['lstm_units'],
                return_sequences=True if _ < self.config['num_lstm_layers'] - 1 else False,
                kernel_regularizer=keras.regularizers.l2(self.config['l2_reg'])
            )(x)
            x = Dropout(self.config['dropout_rate'])(x)
        outputs = Dense(self.num_knots, activation='linear')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
                      loss=self.ql_loss)
        return model

    def ql_loss(self, y_true, y_pred):
        quantile_values = tf.matmul(y_pred, self.M, transpose_b=True)
        e = y_true[:, None] - quantile_values
        tau = tf.constant(self.quantiles, dtype=tf.float32)
        loss = tf.reduce_mean(tf.maximum(tau[None, :] * e, (tau[None, :] - 1) * e), axis=0)
        return tf.reduce_mean(loss)

    def train_model(self, X: np.ndarray, y: np.ndarray, validation_data: Tuple[np.ndarray, np.ndarray], checkpoint_dir: Optional[str] = None):
        current_seed = self.config.get('seed', SEED)
        np.random.seed(current_seed)
        tf.random.set_seed(current_seed)
        random.seed(current_seed)

        try:
            logging.info("Starting model training...")
            X_reshaped, y_train, valid_indices = self.reshape_for_lstm(X, y)
            logging.info(f"X_reshaped shape: {X_reshaped.shape}, y_train shape: {y_train.shape}")

            if self.model is None:
                num_features = X_reshaped.shape[2]
                self.model = self.build_model(num_features)
                logging.info("Model built successfully.")
            else:
                logging.info("Model already built.")

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=self.es_patience, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=self.rlr_factor, patience=self.rlr_patience, min_lr=self.rlr_min_lr, cooldown=self.rlr_cooldown, verbose=1)
            ]

            checkpoint_setup_successful = False
            if checkpoint_dir:
                try:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.weights.h5")
                    callbacks.append(ModelCheckpoint(
                        checkpoint_path,
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=True,
                        save_freq='epoch',
                        verbose=1
                    ))
                    logging.info(f"Checkpointing enabled: Saving best weights to {checkpoint_path}")
                    if os.path.exists(checkpoint_path):
                        self.model.load_weights(checkpoint_path)
                        logging.info(f"Loaded weights from existing checkpoint: {checkpoint_path}")
                    checkpoint_setup_successful = True
                except Exception as e:
                    logging.warning(f"Could not set up checkpointing in train_model: {e}. Continuing without checkpoints.")
                    callbacks = [cb for cb in callbacks if not isinstance(cb, ModelCheckpoint)]
            else:
                logging.info("checkpoint_dir is None. Checkpointing is disabled.")

            X_val, y_val = validation_data
            X_val_reshaped, y_val_train, valid_indices_val = self.reshape_for_lstm(X_val, y_val)
            val_data = (X_val_reshaped, y_val_train)

            logging.info("Starting model.fit...")
            self.history = self.model.fit(
                X_reshaped, y_train,
                epochs=int(float(self.config.get('epochs', 500))),
                batch_size=int(float(self.config.get('batch_size', 64))),
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1
            )
            logging.info("Model.fit completed.")

            train_loss = self.history.history['loss'][-1]
            val_loss = self.history.history.get('val_loss', train_loss)[-1]
            logging.info("Model training completed successfully.")
            return train_loss, val_loss

        except Exception as e:
            logging.error(f"Error in train_model: {str(e)}")
            logging.exception("Traceback for train_model error:")
            raise

    def predict_distribution(self, X: np.ndarray, num_samples: int) -> np.ndarray:
        try:
            np.random.seed(SEED)
            tf.random.set_seed(SEED)
            random.seed(SEED)
            X_reshaped, _, valid_indices = self.reshape_for_lstm(X, np.zeros(X.shape[0]))
            spline_coefficients = self.model.predict(X_reshaped, verbose=0)
            samples = []
            for coeffs in spline_coefficients:
                quantile_values = np.interp(self.quantiles, self.knots, coeffs)
                np.random.seed(SEED)
                sampled_values = np.interp(np.random.uniform(0, 1, num_samples), self.quantiles, quantile_values)
                samples.append(sampled_values)
            return np.array(samples, dtype=np.float32), valid_indices
        except Exception as e:
            logging.error(f"Error in predict_distribution: {str(e)}")
            raise

    def train_calibrator(self, X_val: np.ndarray, y_val: np.ndarray, num_samples_calibration: int = 10000, save_dir: Optional[str] = None):
        try:
            logging.info(f"Training calibrator using {num_samples_calibration} samples for PIT estimation...")
            predictions_val, valid_indices = self.predict_distribution(X_val, num_samples=num_samples_calibration)
            y_val_valid = y_val[valid_indices]
            raw_pit_values = np.array([np.mean(pred <= y) for pred, y in zip(predictions_val, y_val_valid)])
            logging.info(f"Calculated {len(raw_pit_values)} raw PIT values.")

            if save_dir:
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    logging.info(f"Plotting raw PIT vs ranks to {save_dir}")
                    sorted_raw_pit_values = np.sort(raw_pit_values)
                    n_samples_pit = len(sorted_raw_pit_values)
                    uniform_quantiles = (np.arange(n_samples_pit) + 0.5) / n_samples_pit

                    plt.figure(figsize=(8, 8))
                    plt.plot(uniform_quantiles, sorted_raw_pit_values, marker='.', linestyle='none', markersize=4, label='Empirical Raw PIT', color='blue')
                    plt.plot([0, 1], [0, 1], 'k--', label='Ideal Uniform', linewidth=1.5)
                    plt.title('Raw PIT vs. Ranks (Before Calibration)', fontsize=14, fontweight='bold')
                    plt.xlabel('Theoretical Quantiles of Uniform(0,1) (Ranks)', fontsize=12)
                    plt.ylabel('Empirical Quantiles of Raw PIT Values', fontsize=12)
                    plt.legend(fontsize=10)
                    plt.grid(False)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.savefig(os.path.join(save_dir, 'raw_pit_vs_ranks_plot.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    logging.info("Raw PIT vs ranks plot saved.")
                except Exception as plot_e:
                    logging.error(f"Error plotting raw PIT vs ranks: {str(plot_e)}")

            ranks_for_ir = np.linspace(0, 1, len(raw_pit_values))
            self.ir_pit_to_rank = IsotonicRegression(out_of_bounds='clip').fit(raw_pit_values, ranks_for_ir)
            logging.info("ir_pit_to_rank trained.")
            self.ir_rank_to_pit = IsotonicRegression(out_of_bounds='clip').fit(ranks_for_ir, raw_pit_values)
            logging.info("ir_rank_to_pit trained.")
            logging.info("Calibrator training finished.")
        except Exception as e:
            logging.error(f"Error in train_calibrator: {str(e)}")
            logging.exception("Traceback for train_calibrator error:")
            raise

    def predict_distribution_calibrated(self, X: np.ndarray, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        try:
            predictions, valid_indices = self.predict_distribution(X, num_samples)
            if self.ir_pit_to_rank is None or self.ir_rank_to_pit is None:
                raise ValueError("Calibrator not trained. Run train_calibrator first.")
            calibrated_predictions = []
            for pred in predictions:
                pit_values = np.linspace(0, 1, num_samples)
                calibrated_quantiles = self.ir_rank_to_pit.predict(self.ir_pit_to_rank.predict(pit_values))
                calibrated_values = np.interp(pit_values, calibrated_quantiles, pred)
                calibrated_predictions.append(calibrated_values)
            return np.array(calibrated_predictions), valid_indices
        except Exception as e:
            logging.error(f"Error in predict_distribution_calibrated: {str(e)}")
            raise

    def calculate_metrics(self, y_true: np.ndarray, predictions: np.ndarray, scaler=None, naive_error: Optional[float] = None) -> Dict[str, float]:
        try:
            mean_predictions = np.mean(predictions, axis=1)
            q25, q75 = np.percentile(y_true, [25, 75])
            iqr = q75 - q25
            bin_width = 2 * iqr / (len(y_true) ** (1/3))
            data_range = np.max(y_true) - np.min(y_true)
            bins = int(data_range / bin_width) if bin_width > 0 else 1
            y_true_hist, bin_edges = np.histogram(y_true, bins=bins, density=True)
            pred_hist, _ = np.histogram(mean_predictions, bins=bin_edges, density=True)
            
            epsilon = 1e-10
            y_true_hist_smoothed = (y_true_hist + 1) / (np.sum(y_true_hist) + len(y_true_hist))
            pred_hist_smoothed = (pred_hist + 1) / (np.sum(pred_hist) + len(pred_hist))
            
            r2 = r2_score(y_true, mean_predictions)
            nrmse = np.sqrt(np.mean((y_true - mean_predictions) ** 2)) / data_range if data_range != 0 else np.inf
            smape = np.mean(2 * np.abs(y_true - mean_predictions) / (np.abs(y_true) + np.abs(mean_predictions) + epsilon)) * 100
            
            mae = np.mean(np.abs(y_true - mean_predictions))
            mase = np.inf
            if naive_error is not None and not np.isnan(naive_error) and naive_error > 0:
                mase = mae / naive_error
            else:
                logging.warning(f"Skipping MASE calculation: Naive error is invalid ({naive_error}).")

            crps = ps.crps_ensemble(y_true, predictions).mean()
            kl_div = np.sum(y_true_hist_smoothed * np.log(y_true_hist_smoothed / (pred_hist_smoothed + epsilon)))

            quantile_losses = {}
            ql_sum = 0
            if hasattr(self, 'quantiles'):
                for i, q in enumerate(self.quantiles):
                    quantile_pred = np.percentile(predictions, q * 100, axis=1)
                    errors = y_true - quantile_pred
                    pinball_loss = np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors))
                    quantile_losses[f'ql_{int(q*100):02d}'] = pinball_loss
                    ql_sum += pinball_loss
                
                qlm = ql_sum / len(self.quantiles) if len(self.quantiles) > 0 else np.inf
                ql50 = quantile_losses.get('ql_50', np.inf)
                ql90 = quantile_losses.get('ql_90', np.inf)

                predicted_q90_from_samples = np.percentile(predictions, 90, axis=1)
                ql90_specific = np.mean(np.where(y_true - predicted_q90_from_samples >= 0, 0.9 * (y_true - predicted_q90_from_samples), (0.9 - 1) * (y_true - predicted_q90_from_samples)))
                if 'ql_90' not in quantile_losses or np.isinf(ql90):
                    ql90 = ql90_specific
            else:
                logging.warning("Cannot calculate QL metrics: Quantiles attribute not found.")
                ql50 = np.inf
                ql90 = np.inf
                qlm = np.inf

            lower_bound = np.percentile(predictions, 2.5, axis=1)
            upper_bound = np.percentile(predictions, 97.5, axis=1)
            interval_width = upper_bound - lower_bound
            in_sample_mae = np.mean(np.abs(y_true - mean_predictions))
            msis = np.mean(interval_width) / in_sample_mae if in_sample_mae != 0 else np.inf

            pit_values = self._calculate_pit(y_true, predictions)
            pit_mean = np.mean(pit_values)
            pit_std = np.std(pit_values)

            wasserstein = wasserstein_distance(y_true, mean_predictions)

            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, mean_predictions)),
                'r2': r2,
                'nrmse': nrmse,
                'smape': smape,
                'mae': mae,
                'mase': mase,
                'crps': crps,
                'kl_divergence': kl_div,
                'ql50': ql50,
                'ql90': ql90,
                'qlm': qlm,
                'msis': msis,
                'wasserstein': wasserstein,
                'pit_mean': pit_mean,
                'pit_std': pit_std
            }

            if scaler is not None:
                try:
                    y_true_original = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
                    predictions_original = scaler.inverse_transform(predictions.T).T
                    mean_predictions_original = np.mean(predictions_original, axis=1)
                    metrics_original = {
                        'rmse_original': np.sqrt(mean_squared_error(y_true_original, mean_predictions_original)),
                        'r2_original': r2_score(y_true_original, mean_predictions_original),
                        'crps_original': ps.crps_ensemble(y_true_original, predictions_original).mean()
                    }
                    metrics.update(metrics_original)
                except Exception as e:
                    logging.warning(f"Could not calculate metrics on original scale: {e}")
                    metrics['rmse_original'] = float('nan')
                    metrics['r2_original'] = float('nan')
                    metrics['crps_original'] = float('nan')

            return metrics
        except Exception as e:
            logging.error(f"Error in calculate_metrics: {str(e)}")
            logging.exception("Traceback for calculate_metrics error:")
            metric_names = ['rmse', 'r2', 'nrmse', 'smape', 'mae', 'mase', 'crps', 'kl_divergence',
                            'ql50', 'ql90', 'qlm', 'msis', 'wasserstein', 'pit_mean', 'pit_std']
            return {name: float('nan') for name in metric_names}

    def calculate_msis(self, y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.05) -> float:
        lower = np.percentile(y_pred, 100 * alpha / 2, axis=1)
        upper = np.percentile(y_pred, 100 * (1 - alpha / 2), axis=1)
        interval_score = (upper - lower) + (2 / alpha) * (lower - y_true) * (y_true < lower) + (2 / alpha) * (y_true - upper) * (y_true > upper)
        naive_intervals = np.percentile(y_true, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        naive_score = naive_intervals[1] - naive_intervals[0]
        return np.mean(interval_score) / naive_score

    def plot_results(self, y_true: np.ndarray, predictions: np.ndarray, title: str = 'Predictions', save_dir: str = None, scaler=None):
        try:
            if save_dir is None:
                save_dir = os.path.join(r'C:\Users\masoud\results\P\sqp_mlstm', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            os.makedirs(save_dir, exist_ok=True)
            if scaler is not None:
                y_true_original = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
                predictions_original = scaler.inverse_transform(predictions.T).T
                mean_predictions_original = np.mean(predictions_original, axis=1)
                lower_bound_original = np.percentile(predictions_original, 2.5, axis=1)
                upper_bound_original = np.percentile(predictions_original, 97.5, axis=1)
            else:
                y_true_original = y_true
                mean_predictions_original = np.mean(predictions, axis=1) if predictions.ndim > 1 else predictions
                lower_bound_original = np.percentile(predictions, 2.5, axis=1) if predictions.ndim > 1 else None
                upper_bound_original = np.percentile(predictions, 97.5, axis=1) if predictions.ndim > 1 else None
            plt.style.use('seaborn-v0_8-white')
            plt.figure(figsize=(12, 6))
            plt.plot(y_true_original, label='Actual', alpha=0.7, color='blue', linewidth=1.5)
            plt.plot(mean_predictions_original, label='Predicted Mean', alpha=0.7, color='orange', linewidth=1.5)
            if lower_bound_original is not None and upper_bound_original is not None:
                plt.fill_between(range(len(y_true_original)), lower_bound_original, upper_bound_original, alpha=0.3, color='orange', label='95% Prediction Interval')
            plt.title(f'{title} - Spline Quantile Predictions (LSTM)', fontsize=14, fontweight='bold')
            plt.xlabel('Time Step', fontsize=12)
            plt.ylabel('Value (Original Scale)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(False)
            plt.savefig(os.path.join(save_dir, f'{title}_predictions.png'), dpi=300, bbox_inches='tight')
            plt.close()
            if self.history is not None:
                plt.figure(figsize=(12, 6))
                plt.plot(self.history.history['loss'], label='Training Loss', color='blue', linewidth=1.5)
                if 'val_loss' in self.history.history:
                    plt.plot(self.history.history['val_loss'], label='Validation Loss', color='orange', linewidth=1.5)
                plt.title('Model Training History', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(False)
                plt.savefig(os.path.join(save_dir, f'{title}_training_history.png'), dpi=300, bbox_inches='tight')
                plt.close()
            plt.figure(figsize=(12, 6))
            sns.kdeplot(y_true_original, label='Actual Distribution', color='blue', fill=True, alpha=0.3)
            sns.kdeplot(mean_predictions_original, label='Predicted Distribution', color='orange', fill=True, alpha=0.3)
            plt.title(f'{title} - Actual vs Predicted Distributions (Original Scale)', fontsize=14, fontweight='bold')
            plt.xlabel('Value (Original Scale)', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(False)
            plt.savefig(os.path.join(save_dir, f'{title}_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
            plt.figure(figsize=(12, 6))
            prediction_errors = y_true_original - mean_predictions_original
            plt.plot(prediction_errors, label='Prediction Error', color='purple', alpha=0.7, linewidth=1.5)
            plt.title(f'{title} - Prediction Error Over Time (Original Scale)', fontsize=14, fontweight='bold')
            plt.xlabel('Time Step', fontsize=12)
            plt.ylabel('Error (Original Scale)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(False)
            plt.savefig(os.path.join(save_dir, f'{title}_prediction_errors.png'), dpi=300, bbox_inches='tight')
            plt.close()
            if predictions.ndim > 1 and predictions.shape[1] > 0:
                pit_values = self._calculate_pit(y_true, predictions)
                if pit_values is not None and pit_values.shape[0] > 0:
                    plt.figure(figsize=(12, 6))
                    sns.histplot(pit_values, bins=20, kde=True, color='green', alpha=0.5)
                    plt.title(f'{title} - PIT Histogram', fontsize=14, fontweight='bold')
                    plt.xlabel('PIT', fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
                    plt.grid(False)
                    plt.savefig(os.path.join(save_dir, f'{title}_pit_histogram.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            if predictions.ndim > 1 and predictions.shape[0] > 0 and predictions.shape[1] > 0:
                pit_values = self._calculate_pit(y_true, predictions)
                if pit_values is not None and pit_values.shape[0] > 0:
                    plt.figure(figsize=(8, 8))
                    sorted_pit_values = np.sort(pit_values)
                    n_samples = len(sorted_pit_values)
                    uniform_quantiles = (np.arange(n_samples) + 0.5) / n_samples
                    plt.plot(uniform_quantiles, sorted_pit_values, marker='.', linestyle='none', markersize=4, label='Empirical CPIT', color='orange')
                    plt.plot([0, 1], [0, 1], 'k--', label='Ideal Uniform', linewidth=1.5)
                    plt.title(f'{title} - Cumulative PIT Plot', fontsize=14, fontweight='bold')
                    plt.xlabel('Theoretical Quantiles of Uniform(0,1)', fontsize=12)
                    plt.ylabel('Empirical Quantiles of PIT Values', fontsize=12)
                    plt.legend(fontsize=10)
                    plt.grid(False)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.savefig(os.path.join(save_dir, f'{title}_cpit_plot.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            logging.info(f"All plots saved to: {save_dir}")
        except Exception as e:
            logging.error(f"Error in plot_results: {str(e)}")
            raise

    def _calculate_pit(self, y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        try:
            pit_values = np.array([np.mean(pred <= y) for pred, y in zip(predictions, y_true)])
            return pit_values
        except Exception as e:
            logging.error(f"Error in _calculate_pit: {str(e)}")
            return None

def tune_model(config: Dict, checkpoint_dir: Optional[str] = None, naive_error_train: Optional[float] = None):
    current_seed = config.get('seed', SEED)
    np.random.seed(current_seed)
    tf.random.set_seed(current_seed)
    random.seed(current_seed)
    os.environ['PYTHONHASHSEED'] = str(current_seed)

    try:
        logging.info(f"Starting tune_model trial with config: {config}")
        config_converted = {}
        int_params = ['batch_size', 'epochs', 'num_samples', 'time_steps', 'num_knots',
                      'es_patience', 'rlr_patience', 'rlr_cooldown', 'num_lstm_layers', 'lstm_units']
        float_params = ['learning_rate', 'l2_reg', 'rlr_factor', 'rlr_min_lr', 'dropout_rate']

        for key, value in config.items():
            if key in int_params:
                config_converted[key] = int(float(value))
            elif key in float_params:
                config_converted[key] = float(value)
            else:
                config_converted[key] = value

        logging.info("Config converted for model initialization.")
        model = SplineQuantileRegressionModel(config=config_converted)
        logging.info("SplineQuantileRegressionModel instance created within tune_model.")

        train_features = r"C:\Users\masoud\important_features_train_v8_separate_train_test.xlsx"
        train_target = r"C:\Users\masoud\lreturns30w_train_scaled.xlsx"
        test_features = r"C:\Users\masoud\important_features_test_v8_separate_train_test.xlsx"
        test_target = r"C:\Users\masoud\lreturns30w_test_scaled.xlsx"

        X_train, y_train = model.load_data(train_features, train_target)
        X_test, y_test = model.load_data(test_features, test_target)

        X_val, X_final_test_set, y_val, y_final_test_set = train_test_split(
            X_test, y_test, test_size=0.5, random_state=SEED, shuffle=False
        )
        logging.info(f"Data split into Train ({X_train.shape[0]} samples), Validation ({X_val.shape[0]} samples), Final Test ({X_final_test_set.shape[0]} samples).")

        logging.info("Starting base model training within trial...")
        train_loss, val_loss = model.train_model(
            X_train, y_train,
            validation_data=(X_val, y_val),
            checkpoint_dir=checkpoint_dir
        )
        logging.info(f"Base model training finished. Final Train Loss: {train_loss:.4f}, Final Validation Loss: {val_loss:.4f}")

        calibration_sample_size = 200000
        logging.info(f"Starting calibrator training using {calibration_sample_size} samples...")
        model.train_calibrator(X_val, y_val, num_samples_calibration=calibration_sample_size, save_dir=None)
        logging.info("Calibrator training finished.")

        num_samples_eval = config_converted['num_samples']
        scaler_path = r"C:\Users\masoud\robust_scaler.pkl"
        logging.info(f"Loading scaler from: {scaler_path}")
        if not os.path.exists(scaler_path):
            logging.error(f"Scaler file not found: {scaler_path}")
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logging.info("Scaler loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading scaler: {e}")
            raise

        logging.info(f"Starting dynamic adjustment of evaluation sample size from initial {num_samples_eval}...")
        predictions_val, valid_indices_val = model.predict_distribution_calibrated(X_val, num_samples=num_samples_eval)
        y_val_valid = y_val[valid_indices_val]
        metrics = model.calculate_metrics(y_val_valid, predictions_val, scaler=scaler, naive_error=naive_error_train)
        crps_current = metrics.get('crps', float('inf'))
        logging.info(f"  Iteration 1: num_samples_eval={num_samples_eval}, CRPS={crps_current:.4f}")

        crps_prev = float('inf')
        crps_threshold = 0.007
        max_iterations = 5

        for i in range(1, max_iterations):
            if abs(crps_prev - crps_current) < crps_threshold:
                logging.info(f"Converged evaluation sample size at num_samples_eval={num_samples_eval} with CRPS={crps_current:.4f}")
                break
            crps_prev = crps_current
            num_samples_eval *= 2
            max_eval_samples = 50000
            if num_samples_eval > max_eval_samples:
                logging.info(f"Capped evaluation sample size at {max_eval_samples}.")
                num_samples_eval = max_eval_samples
                predictions_val, valid_indices_val = model.predict_distribution_calibrated(X_val, num_samples=num_samples_eval)
                y_val_valid = y_val[valid_indices_val]
                metrics = model.calculate_metrics(y_val_valid, predictions_val, scaler=scaler, naive_error=naive_error_train)
                crps_current = metrics.get('crps', float('inf'))
                break
            predictions_val, valid_indices_val = model.predict_distribution_calibrated(X_val, num_samples=num_samples_eval)
            y_val_valid = y_val[valid_indices_val]
            metrics = model.calculate_metrics(y_val_valid, predictions_val, scaler=scaler, naive_error=naive_error_train)
            crps_current = metrics.get('crps', float('inf'))
            logging.info(f"  Iteration {i+1}: num_samples_eval={num_samples_eval}, CRPS={crps_current:.4f}")

        logging.info(f"Using final num_samples_eval={num_samples_eval} for reporting metrics to Ray Tune.")
        combined_objective = (
            0.4 * metrics.get('crps', 1.0) +
            0.2 * metrics.get('msis', 1.0) +
            0.2 * (1 - metrics.get('r2', 0.0)) +
            0.2 * metrics.get('mae', 1.0)
        )
        logging.info(f"Calculated combined objective: {combined_objective:.4f}")

        report({
            'combined_objective': combined_objective,
            'r2': metrics.get('r2', float('nan')),
            'crps': metrics.get('crps', float('nan')),
            'kl_divergence': metrics.get('kl_divergence', float('nan')),
            'msis': metrics.get('msis', float('nan')),
            'rmse': metrics.get('rmse', float('nan')),
            'mae': metrics.get('mae', float('nan')),
            'mase': metrics.get('mase', float('nan')),
            'pit_mean': metrics.get('pit_mean', float('nan')),
            'pit_std': metrics.get('pit_std', float('nan')),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'num_samples_eval': num_samples_eval,
            'num_samples_calibration_used': calibration_sample_size
        })
        logging.info("Results reported to Ray Tune.")
    except Exception as e:
        logging.error(f"Error in tune_model trial: {str(e)}")
        logging.exception("Traceback for tune_model error:")
        raise

def calculate_ebic(model, X: np.ndarray, y: np.ndarray, train_loss: float, n_features: int, gamma: float = 0.5) -> float:
    n_samples = X.shape[0]
    n_params = sum([w.size for w in model.get_weights()])
    if n_params <= 0:
        raise ValueError("Number of parameters is invalid (zero or negative)")
    log_likelihood = -n_samples * train_loss
    ebic = -2 * log_likelihood + n_params * np.log(n_samples) + 2 * gamma * n_params * np.log(n_features)
    return ebic

# =============================================================================
# 5. EXECUTION SCRIPT
# =============================================================================

def main():
    logging.info("Starting main script.")
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    train_features = r"C:\Users\masoud\important_features_train_v8_separate_train_test.xlsx"
    train_target = r"C:\Users\masoud\lreturns30w_train_scaled.xlsx"
    test_features = r"C:\Users\masoud\important_features_test_v8_separate_train_test.xlsx"
    test_target = r"C:\Users\masoud\lreturns30w_test_scaled.xlsx"
    scaler_path = r"C:\Users\masoud\robust_scaler.pkl"
    EXPERIMENT_NAME = "sqp_lstm_tuning_campaign_v1"

    try:
        temp_config = {
            'seed': SEED,
            'num_lstm_layers': 2,
            'lstm_units': 128,
            'dropout_rate': 0.1,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100,
            'num_knots': 100,
            'l2_reg': 0.001,
            'time_steps': 6,
            'num_samples': 8555,
            'es_patience': 10,
            'rlr_factor': 0.2,
            'rlr_patience': 10,
            'rlr_min_lr': 1e-7,
            'rlr_cooldown': 0
        }
        model_instance = SplineQuantileRegressionModel(config=temp_config)
        logging.info("Temporary SplineQuantileRegressionModel instance created for data loading.")

        X_train_scaled, y_train_scaled = model_instance.load_data(train_features, train_target)
        logging.info(f"Scaled training data loaded: y shape {y_train_scaled.shape}")

        naive_error_train_scaled = np.nan
        if len(y_train_scaled) > 1:
            naive_train_predictions_scaled = y_train_scaled[:-1]
            actual_train_values_scaled = y_train_scaled[1:]
            min_len_scaled = min(len(naive_train_predictions_scaled), len(actual_train_values_scaled))
            naive_train_predictions_scaled = naive_train_predictions_scaled[:min_len_scaled]
            actual_train_values_scaled = actual_train_values_scaled[:min_len_scaled]
            if len(actual_train_values_scaled) > 0:
                naive_error_train_scaled = np.mean(np.abs(actual_train_values_scaled - naive_train_predictions_scaled))
                if naive_error_train_scaled == 0:
                    logging.warning("Naive train error is zero on scaled data. MASE might be inf or nan.")
            else:
                logging.warning("Training data is too short to calculate naive train error after adjustment.")
        else:
            logging.warning("Training data is too short (<=1 sample) to calculate naive train error.")
        logging.info(f"Calculated naive train error (scaled) for MASE: {naive_error_train_scaled}")

        config = {
            'seed': SEED,
            'num_lstm_layers': tune.randint(1, 4),
            'lstm_units': tune.choice([64, 128, 256]),
            'dropout_rate': tune.uniform(0.0, 0.5),
            'learning_rate': tune.loguniform(1e-4, 1e-2),
            'batch_size': tune.choice([32, 64, 128]),
            'epochs': 100,
            'num_knots': tune.randint(80, 300),
            'l2_reg': tune.loguniform(1e-6, 1e-2),
            'time_steps': 6,
            'num_samples': tune.randint(3000, 10000),
            'es_patience': 10,
            'rlr_factor': 0.2,
            'rlr_patience': 10,
            'rlr_min_lr': 1e-7,
            'rlr_cooldown': 0
        }

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, logging_level=logging.INFO)
            logging.info("Ray initialized in main.")
        else:
            logging.info("Ray already initialized.")

        scheduler = ASHAScheduler(
            time_attr='training_iteration',
            max_t=config['epochs'],
            grace_period=5,
            metric="combined_objective",
            mode="min"
        )

        hyperopt_search = HyperOptSearch(
            metric="combined_objective",
            mode="min"
        )

        ray_results_dir = os.path.join(os.path.expanduser("~"), "ray_results_sqp_mlstm")
        os.makedirs(ray_results_dir, exist_ok=True)
        logging.info(f"Ray Tune results will be stored in: {ray_results_dir}")
        logging.info(f"Ray Tune experiment name: {EXPERIMENT_NAME}")

        analysis = tune.run(
            tune.with_parameters(tune_model, naive_error_train=naive_error_train_scaled),
            config=config,
            num_samples=20,
            scheduler=scheduler,
            search_alg=hyperopt_search,
            storage_path=ray_results_dir,
            name=EXPERIMENT_NAME,
            trial_dirname_creator=custom_trial_dirname_creator,
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="combined_objective",
                checkpoint_score_order="min"
            ),
            resume=True
        )
        logging.info("Ray Tune search completed.")

        best_trial = analysis.get_best_trial(metric="combined_objective", mode="min", scope="all")
        if best_trial:
            logging.info(f"Best trial selected: {best_trial.trial_id} from path {best_trial.path}")
            logging.info(f"Best trial config: {best_trial.config}")
            logging.info(f"Best trial last result: {best_trial.last_result}")

            best_config_final = best_trial.config.copy()
            fixed_params_from_initial_config = {
                'seed': SEED, 'time_steps': 6, 'es_patience': 10,
                'rlr_factor': 0.2, 'rlr_patience': 10,
                'rlr_min_lr': 1e-7, 'rlr_cooldown': 0, 'epochs': 100
            }
            for key_fp, val_fp in fixed_params_from_initial_config.items():
                if key_fp not in best_config_final:
                    best_config_final[key_fp] = val_fp

            final_num_samples_eval = best_trial.last_result.get('num_samples_eval', best_config_final.get('num_samples', 8555))
            if isinstance(final_num_samples_eval, float):
                final_num_samples_eval = int(final_num_samples_eval)
            best_config_final['num_samples'] = final_num_samples_eval

            logging.info("SplineQuantileRegressionModel initialized for final training with best config.")
            final_model = SplineQuantileRegressionModel(config=best_config_final)

            final_train_features_path = r"C:\Users\masoud\important_features_train_v8_separate_train_test.xlsx"
            final_train_target_path = r"C:\Users\masoud\lreturns30w_train_scaled.xlsx"
            X_final_train, y_final_train = final_model.load_data(final_train_features_path, final_train_target_path)
            logging.info(f"Full training data loaded: X shape {X_final_train.shape}, y shape {y_final_train.shape}")

            final_test_features_path = r"C:\Users\masoud\important_features_test_v8_separate_train_test.xlsx"
            final_test_target_path = r"C:\Users\masoud\lreturns30w_test_scaled.xlsx"
            X_final_test_set_and_val, y_final_test_set_and_val = final_model.load_data(final_test_features_path, final_test_target_path)
            
            X_val_final, X_final_test_set, y_val_final, y_final_test_set = train_test_split(
                X_final_test_set_and_val, y_final_test_set_and_val, test_size=0.5, random_state=SEED, shuffle=False
            )
            logging.info(f"Data split for final evaluation: Validation ({X_val_final.shape[0]} samples), Final Test ({X_final_test_set.shape[0]} samples).")

            logging.info("Starting final model training with best hyperparameters...")
            final_model_checkpoint_dir = os.path.join(best_trial.path, "final_model_checkpoint")
            os.makedirs(final_model_checkpoint_dir, exist_ok=True)

            final_model_train_loss, final_model_val_loss = final_model.train_model(
                X_final_train, y_final_train,
                validation_data=(X_val_final, y_val_final),
                checkpoint_dir=final_model_checkpoint_dir
            )
            logging.info(f"Final model training finished. Train Loss: {final_model_train_loss:.4f}, Val Loss: {final_model_val_loss:.4f}")

            calibration_sample_size_final_val = 200000
            logging.info(f"Training final calibrator with {calibration_sample_size_final_val} samples...")
            base_save_dir_plots = r'C:\Users\masoud\results\P\sqp_lstm_final'
            os.makedirs(base_save_dir_plots, exist_ok=True)
            current_run_timestamp_final = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_model_plot_save_dir_path = os.path.join(base_save_dir_plots, f"run_{EXPERIMENT_NAME}_{best_trial.trial_id[:8]}_{current_run_timestamp_final}")
            os.makedirs(final_model_plot_save_dir_path, exist_ok=True)
            logging.info(f"Final results and plots will be saved in: {final_model_plot_save_dir_path}")

            final_model.train_calibrator(X_val_final, y_val_final, num_samples_calibration=calibration_sample_size_final_val, save_dir=final_model_plot_save_dir_path)
            logging.info("Calibrator training finished.")

            logging.info(f"Loading scaler from: {scaler_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            with open(scaler_path, 'rb') as f_scaler:
                scaler_loaded_final = pickle.load(f_scaler)
            logging.info("Scaler loaded successfully.")

            logging.info(f"Evaluating final model on validation set using {final_num_samples_eval:.0f} samples...")
            predictions_val_final_cal, valid_indices_val_final_dt = final_model.predict_distribution_calibrated(X_val_final, num_samples=int(final_num_samples_eval))
            y_val_final_valid_dt = y_val_final[valid_indices_val_final_dt]
            metrics_val_final_res = final_model.calculate_metrics(y_val_final_valid_dt, predictions_val_final_cal, scaler=scaler_loaded_final, naive_error=naive_error_train_scaled)
            logging.info("Final validation set performance (calibrated):")
            for metric_key, metric_val in metrics_val_final_res.items():
                if isinstance(metric_val, (int, float)) and not np.isnan(metric_val):
                    logging.info(f"Validation {metric_key}: {metric_val:.4f}")
                else:
                    logging.info(f"Validation {metric_key}: {metric_val}")

            final_model.plot_results(y_val_final_valid_dt, predictions_val_final_cal, title='Final Validation Set Predictions (Calibrated)', scaler=scaler_loaded_final, save_dir=final_model_plot_save_dir_path)

            logging.info(f"Evaluating final model on test set using {final_num_samples_eval:.0f} samples...")
            predictions_test_final_cal, valid_indices_test_final_dt = final_model.predict_distribution_calibrated(X_final_test_set, num_samples=int(final_num_samples_eval))
            y_test_final_valid_dt = y_final_test_set[valid_indices_test_final_dt]
            metrics_test_final_res = final_model.calculate_metrics(y_test_final_valid_dt, predictions_test_final_cal, scaler=scaler_loaded_final, naive_error=naive_error_train_scaled)
            logging.info("Final test set performance (calibrated):")
            for metric_key, metric_val in metrics_test_final_res.items():
                if isinstance(metric_val, (int, float)) and not np.isnan(metric_val):
                    logging.info(f"Test {metric_key}: {metric_val:.4f}")
                else:
                    logging.info(f"Test {metric_key}: {metric_val}")

            final_model.plot_results(y_test_final_valid_dt, predictions_test_final_cal, title='Final Test Set Predictions (Calibrated)', scaler=scaler_loaded_final, save_dir=final_model_plot_save_dir_path)

            if final_model.model:
                final_model_path_to_save_final = os.path.join(final_model_plot_save_dir_path, 'sqp_mlstm_model_final.keras')
                final_model.model.save(final_model_path_to_save_final)
                logging.info(f"Final model saved at {final_model_path_to_save_final}")
            else:
                logging.error("Final model object was not created, cannot save.")
        else:
            logging.error(f"Ray Tune analysis for '{EXPERIMENT_NAME}' did not return a best trial. Check logs in {ray_results_dir}/{EXPERIMENT_NAME}")

    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
    except FileNotFoundError as e_fnf_main:
        logging.error(f"Data or scaler file not found in main: {e_fnf_main}")
    except ValueError as e_val_main:
        logging.error(f"Value error occurred in main: {e_val_main}")
    except Exception as e_generic_main:
        logging.exception(f"An unexpected error occurred in main: {str(e_generic_main)}")
    finally:
        if ray.is_initialized():
            ray.shutdown()
            logging.info("Ray shut down.")

if __name__ == "__main__":
    main()
