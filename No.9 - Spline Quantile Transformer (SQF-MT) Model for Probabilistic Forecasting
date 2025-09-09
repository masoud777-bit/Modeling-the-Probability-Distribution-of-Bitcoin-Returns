import warnings
import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional
from datetime import datetime
import random
from scipy.stats import wasserstein_distance, gaussian_kde
from sklearn.isotonic import IsotonicRegression
import pickle
import properscoring as ps
from scipy import stats
import xlsxwriter

# Configuration & Setup
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.WARNING)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(log_dir, f'spline_quantile_transformer_final_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)

# best confing from No.7 - Spline Quantile Function Multi Transformer (SQF-MT) Model
best_config = {
    'seed': 42, 'num_encoder_layers': 2, 'num_heads': 8, 'key_dim': 64, 'ffn_units': 128,
    'dropout_rate': 0.26726798143509267, 'learning_rate': 0.009536474386655733, 'batch_size': 32,
    'epochs': 100, 'num_knots': 291, 'l2_reg': 0.00013217157013413488, 'time_steps': 6,
    'num_samples': 7962, 'es_patience': 10, 'rlr_factor': 0.2, 'rlr_patience': 10,
    'rlr_min_lr': 1e-07, 'rlr_cooldown': 0
}

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
        self.history = None
        self.config = config
        self.es_patience = int(float(config.get('es_patience', 10)))
        self.rlr_factor = float(config.get('rlr_factor', 0.2))
        self.rlr_patience = int(float(config.get('rlr_patience', 10)))
        self.rlr_min_lr = float(config.get('rlr_min_lr', 1e-7))
        self.rlr_cooldown = int(float(config.get('rlr_cooldown', 0)))

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

    def reshape_for_transformer(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def get_positional_encoding(self, seq_len, d_model):
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        return tf.constant(pe, dtype=tf.float32)

    def encoder_layer(self, units, d_model, num_heads, key_dim, dropout_rate):
        inputs = tf.keras.Input(shape=(None, d_model))
        attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
        attention = tf.keras.layers.Dropout(dropout_rate)(attention)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
        outputs = tf.keras.layers.Dense(units, activation='relu', kernel_regularizer=keras.regularizers.l2(self.config['l2_reg']))(attention)
        outputs = tf.keras.layers.Dense(d_model, kernel_regularizer=keras.regularizers.l2(self.config['l2_reg']))(outputs)
        outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def build_model(self, num_features: int) -> Model:
        inputs = Input(shape=(self.time_steps, num_features))
        pos_encoding = self.get_positional_encoding(self.time_steps, num_features)
        x = inputs + pos_encoding
        for _ in range(self.config['num_encoder_layers']):
            x = self.encoder_layer(
                units=self.config['ffn_units'],
                d_model=num_features,
                num_heads=self.config['num_heads'],
                key_dim=self.config['key_dim'],
                dropout_rate=self.config['dropout_rate']
            )(x)
        last_step = x[:, -1, :]
        outputs = Dense(self.num_knots, activation='linear', kernel_regularizer=keras.regularizers.l2(self.config['l2_reg']))(last_step)
        model = Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']), loss=self.ql_loss)
        return model

    def ql_loss(self, y_true, y_pred):
        quantile_values = tf.matmul(y_pred, self.M, transpose_b=True)
        e = y_true[:, None] - quantile_values
        tau = tf.constant(self.quantiles, dtype=tf.float32)
        loss = tf.reduce_mean(tf.maximum(tau[None, :] * e, (tau[None, :] - 1) * e), axis=0)
        return tf.reduce_mean(loss)

    def train_model(self, X: np.ndarray, y: np.ndarray, validation_data: Tuple[np.ndarray, np.ndarray], checkpoint_dir: Optional[str] = None):
        np.random.seed(self.config.get('seed', SEED))
        tf.random.set_seed(self.config.get('seed', SEED))
        random.seed(self.config.get('seed', SEED))
        try:
            X_reshaped, y_train, valid_indices = self.reshape_for_transformer(X, y)
            logging.info(f"X_reshaped shape: {X_reshaped.shape}, y_train shape: {y_train.shape}")
            if self.model is None:
                num_features = X_reshaped.shape[2]
                self.model = self.build_model(num_features)
                logging.info("Model built successfully.")
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=self.es_patience, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=self.rlr_factor, patience=self.rlr_patience, min_lr=self.rlr_min_lr, cooldown=self.rlr_cooldown, verbose=1)
            ]
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.weights.h5")
                callbacks.append(ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                ))
                if os.path.exists(checkpoint_path):
                    self.model.load_weights(checkpoint_path)
                    logging.info(f"Loaded weights from existing checkpoint: {checkpoint_path}")
            X_val, y_val = validation_data
            X_val_reshaped, y_val_train, _ = self.reshape_for_transformer(X_val, y_val)
            self.history = self.model.fit(
                X_reshaped, y_train,
                epochs=int(float(self.config.get('epochs', 100))),
                batch_size=int(float(self.config.get('batch_size', 32))),
                validation_data=(X_val_reshaped, y_val_train),
                callbacks=callbacks,
                verbose=1
            )
            train_loss = self.history.history['loss'][-1]
            val_loss = self.history.history.get('val_loss', train_loss)[-1]
            logging.info("Model training completed successfully.")
            return train_loss, val_loss
        except Exception as e:
            logging.error(f"Error in train_model: {str(e)}")
            raise

    def predict_distribution(self, X: np.ndarray, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        random.seed(SEED)
        X_reshaped, _, valid_indices = self.reshape_for_transformer(X, np.zeros(X.shape[0]))
        spline_coefficients = self.model.predict(X_reshaped, verbose=0)
        samples = []
        for coeffs in spline_coefficients:
            quantile_values = np.interp(self.quantiles, self.knots, coeffs)
            sampled_values = np.interp(np.random.uniform(0, 1, num_samples), self.quantiles, quantile_values)
            samples.append(sampled_values)
        return np.array(samples, dtype=np.float32), valid_indices

    def train_calibrator(self, X_val: np.ndarray, y_val: np.ndarray, num_samples_calibration: int, save_dir: Optional[str] = None):
        predictions_val, valid_indices = self.predict_distribution(X_val, num_samples=num_samples_calibration)
        y_val_valid = y_val[valid_indices]
        raw_pit_values = np.mean(predictions_val <= y_val_valid[:, None], axis=1)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
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
        ranks_for_ir = np.linspace(0, 1, len(raw_pit_values))
        self.ir_pit_to_rank = IsotonicRegression(out_of_bounds='clip').fit(raw_pit_values, ranks_for_ir)
        self.ir_rank_to_pit = IsotonicRegression(out_of_bounds='clip').fit(ranks_for_ir, raw_pit_values)
        logging.info("Calibrator training finished.")

    def predict_distribution_calibrated(self, X: np.ndarray, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
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

    def plot_results(self, y_true: np.ndarray, predictions: np.ndarray, title: str = 'Predictions', save_dir: str = None, scaler=None):
        try:
            if save_dir is None:
                save_dir = r"C:\Users\masoud\results\P\sqp_transformer_final\final"
            os.makedirs(save_dir, exist_ok=True)
            if scaler is not None:
                y_true_original = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
                predictions_original = scaler.inverse_transform(predictions.T).T
                mean_predictions_original = np.mean(predictions_original, axis=1)
                lower_bound_original = np.percentile(predictions_original, 2.5, axis=1)
                upper_bound_original = np.percentile(predictions_original, 97.5, axis=1)
            else:
                y_true_original = y_true
                mean_predictions_original = np.mean(predictions, axis=1)
                lower_bound_original = np.percentile(predictions, 2.5, axis=1)
                upper_bound_original = np.percentile(predictions, 97.5, axis=1)
            # Plot Predicted Distribution with actual values for comparison
            plt.style.use('seaborn-v0_8-white')
            plt.figure(figsize=(12, 6))
            plt.plot(y_true_original, label='Actual Values', alpha=0.7, color='blue', linewidth=1.5)
            plt.plot(mean_predictions_original, label='Predicted Mean', alpha=0.7, color='orange', linewidth=1.5)
            plt.fill_between(range(len(mean_predictions_original)), lower_bound_original, upper_bound_original, alpha=0.3, color='gray', label='95% Prediction Interval')
            plt.xlabel('Time Step', fontsize=12)
            plt.ylabel('Value (Original Scale)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(False)
            plt.savefig(os.path.join(save_dir, f'{title}_predicted_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            if self.history is not None:
                plt.figure(figsize=(12, 6))
                plt.plot(self.history.history['loss'], label='Training Loss', color='blue', linewidth=1.5)
                if 'val_loss' in self.history.history:
                    plt.plot(self.history.history['val_loss'], label='Validation Loss', color='orange', linewidth=1.5)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(False)
                plt.savefig(os.path.join(save_dir, f'{title}_training_history.png'), dpi=300, bbox_inches='tight')
                plt.close()
                plt.figure(figsize=(12, 6))
                sns.kdeplot(mean_predictions_original, label='Predicted Distribution', color='black', fill=False, linewidth=2)
                plt.xlabel('Value (Original Scale)', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(False)
                plt.savefig(os.path.join(save_dir, f'{title}_predicted_distribution_kde.png'), dpi=300, bbox_inches='tight')
                plt.close()
            logging.info(f"Plots saved to: {save_dir}")
        except Exception as e:
            logging.error(f"Error in plot_results: {str(e)}")
            raise

    def calculate_var(self, data: np.ndarray, confidence_level: float, is_left_tail: bool = True) -> float:
        if len(data) == 0:
            return np.nan
        if is_left_tail:
            return np.percentile(data, 100 * confidence_level)
        else:
            return np.percentile(data, 100 * (1 - confidence_level))

    def calculate_cvar(self, data: np.ndarray, confidence_level: float, is_left_tail: bool = True) -> float:
        data = data[~np.isnan(data)]
        if len(data) == 0:
            return np.nan
        var = self.calculate_var(data, confidence_level, is_left_tail)
        if is_left_tail:
            tail_losses = data[data <= var]
            if len(tail_losses) == 0:
                return var
            return np.mean(tail_losses)
        else:
            tail_losses = data[data >= var]
            if len(tail_losses) == 0:
                return var
            return np.mean(tail_losses)

    def calculate_glue_var(self, data: np.ndarray, alpha: float, beta: float, omega1: float, omega2: float, is_left_tail: bool = True) -> float:
        cvar_alpha = self.calculate_cvar(data, alpha, is_left_tail)
        cvar_beta = self.calculate_cvar(data, beta, is_left_tail)
        var_alpha = self.calculate_var(data, alpha, is_left_tail)
        if omega1 + omega2 > 1:
            omega1, omega2 = omega1 / (omega1 + omega2), omega2 / (omega1 + omega2)
        return omega1 * cvar_beta + omega2 * cvar_alpha + (1 - omega1 - omega2) * var_alpha

    def improved_kl_divergence(self, p, q, method='simpson'):
        epsilon = 1e-10
        mask = (p > epsilon) & (q > epsilon)
        p_clean = p[mask]
        q_clean = q[mask]
        if len(p_clean) == 0:
            return np.inf
        kl = p_clean * np.log(p_clean / q_clean)
        if method == 'simpson':
            from scipy.integrate import simpson
            dx = 1.0 / len(kl)
            return simpson(kl, dx=dx)
        else:
            return np.sum(kl) / len(kl)

    def calculate_aic_bic(self, loglik, n_params, n_samples):
        aic = -2 * loglik + 2 * n_params
        bic = -2 * loglik + n_params * np.log(n_samples)
        return aic, bic

    def safe_pdf_calculation(self, dist, x, params, name):
        try:
            if name.startswith('t-df='):
                df = int(name.split('=')[1])
                if len(params) >= 2:
                    loc, scale = params[0], params[1]
                    return stats.t(df, loc, scale).pdf(x)
                else:
                    return stats.t(df).pdf(x)
            elif name == 'beta':
                if len(params) == 4:
                    a, b, loc, scale = params
                    if a > 0 and b > 0 and scale > 0:
                        return stats.beta(a, b, loc, scale).pdf(x)
                return None
            elif name == 'gamma':
                if len(params) >= 2:
                    a, loc, scale = params[0], params[1] if len(params) > 2 else 0, params[-1]
                    if a > 0 and scale > 0:
                        return stats.gamma(a, loc, scale).pdf(x)
                return None
            else:
                return dist.pdf(x, *params)
        except Exception as e:
            logging.warning(f"Error calculating PDF for {name}: {e}")
            return None

    def safe_logpdf_calculation(self, dist, data, params, name):
        try:
            if name.startswith('t-df='):
                df = int(name.split('=')[1])
                if len(params) >= 2:
                    loc, scale = params[0], params[1]
                    logpdf_values = stats.t(df, loc, scale).logpdf(data)
                else:
                    logpdf_values = stats.t(df).logpdf(data)
            elif name == 'beta':
                if len(params) == 4:
                    a, b, loc, scale = params
                    if a > 0 and b > 0 and scale > 0:
                        logpdf_values = stats.beta(a, b, loc, scale).logpdf(data)
                    else:
                        return None
                else:
                    return None
            elif name == 'gamma':
                if len(params) >= 2:
                    a, loc, scale = params[0], params[1] if len(params) > 2 else 0, params[-1]
                    if a > 0 and scale > 0:
                        logpdf_values = stats.gamma(a, loc, scale).logpdf(data)
                    else:
                        return None
                else:
                    return None
            else:
                logpdf_values = dist.logpdf(data, *params)
            if np.any(np.isinf(logpdf_values)) or np.any(np.isnan(logpdf_values)):
                return None
            return np.sum(logpdf_values)
        except Exception as e:
            logging.warning(f"Error calculating log-PDF for {name}: {e}")
            return None

    def get_param_count(self, name, params):
        param_counts = {
            'normal': 2, 'laplace': 2, 'logistic': 2, 'gumbel': 2, 'gumbel_min': 2,
            'expon': 1, 'rayleigh': 1, 'chi2': 1,
            'gamma': 2, 'weibull': 2, 'lognormal': 2, 'inverse_gaussian': 2,
            'beta': 2
        }
        if name.startswith('t-df='):
            return 2
        return param_counts.get(name, len(params) if params else 0)

    def ahp_weights(self, criteria):
        """
        Determine weights for criteria using the Analytic Hierarchy Process (AHP).

        Args:
            criteria (list): List of criteria names.

        Returns:
            dict: A dictionary with criteria names as keys and their weights as values.
        """
        n = len(criteria)
        # Example pairwise comparison matrix (you should customize this based on your domain knowledge)
        # This is a placeholder; you should replace it with actual pairwise comparisons
        comparison_matrix = np.array([
            [1, 2, 2, 4], # KL vs others
            [1/2, 1, 1, 2], # AIC vs others
            [1/2, 1, 1, 2], # BIC vs others
            [1/4, 1/2, 1/2, 1] # KS vs others
        ])
        # Normalize the comparison matrix
        normalized_matrix = comparison_matrix / comparison_matrix.sum(axis=0, keepdims=True)
        # Calculate the priority vector (weights)
        weights = normalized_matrix.mean(axis=1)
        # Normalize the weights to sum to 1
        weights = weights / weights.sum()
        # Create a dictionary of criteria and their weights
        criteria_weights = {criteria[i]: weights[i] for i in range(n)}
        return criteria_weights

    def weights_dict_to_str(self, metric_names, weight_values):
        """Helper function to create a string representation of metric weights."""
        return ", ".join([f"{name}: {val:.3f}" for name, val in zip(metric_names, weight_values)])

    def signed_log1p(self, x):
        """Apply signed log1p transformation to an array."""
        return np.sign(x) * np.log1p(np.abs(x))

    def topsis_ranking(self, results, criteria_metrics, criteria_types, criteria_weights_dict=None, num_top=5, log_scale_scores=False):
        """
        Rank distributions using the TOPSIS method.

        Args:
            results (dict): Dictionary containing distribution names as keys and their metric scores.
            criteria_metrics (list): List of metric keys used from the results dictionary.
            criteria_types (list): List indicating whether each criterion is 'cost' (lower is better) or 'benefit' (higher is better).
            criteria_weights_dict (dict, optional): Dictionary of weights for each criterion.
            num_top (int): Number of top distributions to report.
            log_scale_scores (bool): Whether to apply signed_log1p transformation to the scores.

        Returns:
            list: List of names of the top N distributions.
        """
        if not results:
            logging.warning("TOPSIS: No results available for ranking.")
            return []

        # 1. Filter distributions and build decision matrix
        valid_distributions_names = []
        decision_matrix_list = []

        for name, res_data in results.items():
            is_valid = True
            current_row = []
            if not res_data:
                is_valid = False
            else:
                for metric_key in criteria_metrics:
                    if metric_key not in res_data or res_data[metric_key] is None or \
                       np.isnan(res_data[metric_key]) or np.isinf(res_data[metric_key]):
                        is_valid = False
                        logging.debug(f"TOPSIS: Distribution {name} excluded due to invalid value for {metric_key}.")
                        break
                    current_row.append(res_data[metric_key])
            if is_valid:
                valid_distributions_names.append(name)
                decision_matrix_list.append(current_row)

        if not valid_distributions_names:
            logging.warning("TOPSIS: No distributions have valid scores for all specified metrics.")
            return []

        num_alternatives = len(valid_distributions_names)
        num_criteria = len(criteria_metrics)
        decision_matrix = np.array(decision_matrix_list, dtype=float)

        if num_alternatives == 0:  # Should be caught by previous check
            logging.warning("TOPSIS: Number of valid distributions is zero.")
            return []

        # Apply signed_log1p transformation if specified
        if log_scale_scores:
            decision_matrix = self.signed_log1p(decision_matrix)

        # Assign weights
        weights = np.zeros(num_criteria)
        if criteria_weights_dict is None:
            logging.info("TOPSIS: Using equal weights for criteria.")
            weights.fill(1 / num_criteria)
        else:
            current_sum = 0
            for i, key in enumerate(criteria_metrics):
                weights[i] = criteria_weights_dict.get(key, 0)  # Default to 0 if weight not specified
                current_sum += weights[i]
            if not np.isclose(current_sum, 1.0):
                logging.warning(f"TOPSIS: Sum of provided weights ({current_sum:.3f}) is not 1. Weights will be normalized.")
                if current_sum == 0:  # If all weights are zero, use equal weights
                    logging.warning("TOPSIS: All provided weights are zero, using equal weights.")
                    weights.fill(1 / num_criteria)
                else:
                    weights = weights / current_sum
            elif any(w < 0 for w in weights):
                logging.error("TOPSIS: Negative weights are not allowed. Please adjust the weights.")
                return []

        # 2. Normalize the decision matrix (vector normalization)
        norm_decision_matrix = np.zeros_like(decision_matrix)
        for j in range(num_criteria):
            col_sum_sq = np.sqrt(np.sum(decision_matrix[:, j]**2))
            if col_sum_sq == 0:
                norm_decision_matrix[:, j] = 0  # If all values in the column are zero
            else:
                norm_decision_matrix[:, j] = decision_matrix[:, j] / col_sum_sq

        # 3. Construct the weighted normalized decision matrix
        weighted_norm_matrix = norm_decision_matrix * weights

        # 4. Determine the positive ideal solution (A_star) and negative ideal solution (A_minus)
        A_star = np.zeros(num_criteria)
        A_minus = np.zeros(num_criteria)
        for j in range(num_criteria):
            if criteria_types[j].lower() == 'cost':  # Lower is better
                A_star[j] = np.min(weighted_norm_matrix[:, j])
                A_minus[j] = np.max(weighted_norm_matrix[:, j])
            elif criteria_types[j].lower() == 'benefit':  # Higher is better
                A_star[j] = np.max(weighted_norm_matrix[:, j])
                A_minus[j] = np.min(weighted_norm_matrix[:, j])
            else:
                logging.error(f"TOPSIS: Invalid criterion type '{criteria_types[j]}' for metric '{criteria_metrics[j]}'. Must be 'cost' or 'benefit'.")
                return []

        # 5. Calculate the distance from the positive ideal solution and negative ideal solution (Euclidean distance)
        S_star = np.zeros(num_alternatives)
        S_minus = np.zeros(num_alternatives)
        for i in range(num_alternatives):
            S_star[i] = np.sqrt(np.sum((weighted_norm_matrix[i, :] - A_star)**2))
            S_minus[i] = np.sqrt(np.sum((weighted_norm_matrix[i, :] - A_minus)**2))

        # 6. Calculate the relative closeness to the ideal solution (C_star)
        sum_S = S_star + S_minus
        C_star = np.zeros(num_alternatives)
        for i in range(num_alternatives):
            if sum_S[i] == 0:
                # This case occurs when an option is simultaneously the positive and negative ideal (e.g., if there's only one option or all options are identical for a criterion).
                # If S_star[i] is zero (completely ideal), C_star is set to 1.
                C_star[i] = 1.0 if S_star[i] == 0 else 0.0
            else:
                C_star[i] = S_minus[i] / sum_S[i]

        # 7. Rank the alternatives (higher C_star is better)
        ranked_alternatives = sorted(zip(valid_distributions_names, C_star, S_star, S_minus, range(num_alternatives)),
                                     key=lambda x: x[1], reverse=True)

        # Display results in log
        num_top_actual = min(num_top, num_alternatives)
        if num_alternatives < num_top:
            logging.warning(f"TOPSIS: Number of valid distributions ({num_alternatives}) is less than requested ({num_top}). Showing all {num_alternatives} valid distributions.")
        logging.info("\n" + "="*70)
        logging.info(f"Distribution Ranking using TOPSIS Method (Top {num_top_actual}):")
        logging.info(f" Metrics used: {criteria_metrics}")
        logging.info(f" Metric types (cost/benefit): {criteria_types}")
        logging.info(f" Metric weights: {self.weights_dict_to_str(criteria_metrics, weights)}")
        logging.info("="*70)
        header = f"{'Rank':<6} {'Distribution':<20} {'TOPSIS Score (C*)':<20} {'S* (Distance from Ideal+)':<25} {'S- (Distance from Ideal-)':<25}"
        logging.info(header)
        logging.info("-" * len(header))
        top_n_names = []
        for i, (name, score, s_plus_val, s_minus_val, original_idx) in enumerate(ranked_alternatives[:num_top_actual]):
            top_n_names.append(name)
            raw_values_str = ", ".join([f"{val:.4g}" for val in decision_matrix[original_idx,:]])
            log_line = (f"{i+1:<6} {name:<20} {score:<20.4f} {s_plus_val:<25.4f} {s_minus_val:<25.4f}")
            logging.info(log_line)
            logging.debug(f" Raw values for {name}: [{raw_values_str}]")
        logging.info("="*70)

        return top_n_names

    def improved_distribution_comparison(self, mean_predictions_unscaled, output_dir):
        logging.info("Starting improved distribution comparison")

        # تعریف توزیع‌ها
        distributions = {
            'chi2': stats.chi2,
            'expon': stats.expon,
            'genextreme': stats.genextreme,
            'rayleigh': stats.rayleigh,
            'gumbel_r': stats.gumbel_r,
            'cauchy': stats.cauchy,
            'laplace': stats.laplace,
            'gumbel_l': stats.gumbel_l,
            'invgauss': stats.invgauss,
            'hypsecant': stats.hypsecant,
            'logistic': stats.logistic,
            'weibull_min': stats.weibull_min,
            'gamma': stats.gamma,
            'johnsonsu': stats.johnsonsu,
            'normal': stats.norm,
            'lognormal': stats.lognorm,
            'gennorm': stats.gennorm,
            't-df=3': stats.t,
            't-df=7': stats.t,
            'uniform': stats.uniform  # Adding uniform distribution to make it 20
        }

        # برازش توزیع‌ها
        params = {}
        valid_distributions = {}
        for name, dist in distributions.items():
            try:
                if name.startswith('t-df='):
                    df = int(name.split('=')[1])
                    current_params = dist.fit(mean_predictions_unscaled, fdf=df)
                    params[name] = current_params
                    valid_distributions[name] = dist
                elif name == 'beta':
                    data_min, data_max = np.min(mean_predictions_unscaled), np.max(mean_predictions_unscaled)
                    data_range = data_max - data_min
                    if data_range > 0:
                        normalized_data = (mean_predictions_unscaled - data_min) / data_range
                        normalized_data = np.clip(normalized_data, 1e-6, 1-1e-6)
                        fit_params = stats.beta.fit(normalized_data, floc=0, fscale=1)
                        params[name] = fit_params
                        valid_distributions[name] = dist
                    else:
                        logging.warning(f"Cannot fit {name}: data has zero range")
                elif name == 'lognormal':
                    if np.all(mean_predictions_unscaled > 0):
                        params[name] = dist.fit(mean_predictions_unscaled)
                        valid_distributions[name] = dist
                    else:
                        shifted_data = mean_predictions_unscaled - np.min(mean_predictions_unscaled) + 1e-6
                        if np.all(shifted_data > 0):
                            try:
                                params[name] = dist.fit(mean_predictions_unscaled)
                                valid_distributions[name] = dist
                            except Exception as e_lognorm:
                                logging.warning(f"Cannot fit {name} even with shifting attempt: {e_lognorm}")
                        else:
                            logging.warning(f"Cannot fit {name}: data contains non-positive values and shifting failed.")
                else:
                    params[name] = dist.fit(mean_predictions_unscaled)
                    valid_distributions[name] = dist
            except Exception as e:
                logging.warning(f"Could not fit {name}: {e}")

        # محاسبه معیارها
        n_points = 1000
        data_min, data_max = np.min(mean_predictions_unscaled), np.max(mean_predictions_unscaled)
        margin = (data_max - data_min) * 0.1
        x = np.linspace(data_min - margin, data_max + margin, n_points)

        # محاسبه KDE مرجع
        kde = gaussian_kde(mean_predictions_unscaled)
        kde.set_bandwidth(kde.factor * 0.8)
        y_true = kde(x)
        y_true = y_true / np.trapz(y_true, x)

        results = {}
        n_samples = len(mean_predictions_unscaled)

        for name in valid_distributions.keys():
            if params[name] is not None:
                y_pred = self.safe_pdf_calculation(valid_distributions[name], x, params[name], name)
                if y_pred is not None:
                    y_pred = y_pred / np.trapz(y_pred, x)
                    kl_div = self.improved_kl_divergence(y_true, y_pred, method='simpson')
                    loglik = self.safe_logpdf_calculation(valid_distributions[name], mean_predictions_unscaled, params[name], name)
                    if loglik is not None:
                        n_params = self.get_param_count(name, params[name])
                        aic, bic = self.calculate_aic_bic(loglik, n_params, n_samples)
                        try:
                            if name.startswith('t-df='):
                                df = int(name.split('=')[1])
                                loc, scale = params[name][:2]  # Ensure only loc and scale are used
                                ks_stat, ks_p = stats.kstest(mean_predictions_unscaled, lambda x: stats.t(df, loc, scale).cdf(x))
                            elif name == 'beta':
                                a, b, loc, scale = params[name]
                                ks_stat, ks_p = stats.kstest(mean_predictions_unscaled, lambda x: stats.beta(a, b, loc, scale).cdf(x))
                            elif name == 'gamma':
                                a, loc, scale = params[name][0], params[name][1] if len(params[name]) > 1 else 0, params[name][-1]
                                ks_stat, ks_p = stats.kstest(mean_predictions_unscaled, lambda x: stats.gamma(a, loc, scale).cdf(x))
                            else:
                                ks_stat, ks_p = stats.kstest(mean_predictions_unscaled, lambda x: valid_distributions[name].cdf(x, *params[name]))
                        except Exception as e:
                            ks_stat, ks_p = np.nan, np.nan
                            logging.warning(f"Could not calculate KS statistics for {name}: {e}")
                        results[name] = {
                            'kl_divergence': kl_div,
                            'aic': aic,
                            'bic': bic,
                            'loglikelihood': loglik,
                            'n_params': n_params,
                            'ks_statistic': ks_stat,
                            'ks_pvalue': ks_p,
                            'y_pred': y_pred,
                            'params': params[name]
                        }

        # مرتب‌سازی نتایج
        if results:
            kl_ranking = sorted([(name, res['kl_divergence']) for name, res in results.items() if not np.isinf(res['kl_divergence'])], key=lambda x: x[1])
            aic_ranking = sorted([(name, res['aic']) for name, res in results.items()], key=lambda x: x[1])
            bic_ranking = sorted([(name, res['bic']) for name, res in results.items()], key=lambda x: x[1])
            ks_ranking = sorted([(name, res['ks_statistic']) for name, res in results.items() if not np.isnan(res['ks_statistic'])], key=lambda x: x[1])

            # نمایش نتایج
            logging.info("\n" + "="*50)
            logging.info("IMPROVED DISTRIBUTION COMPARISON RESULTS")
            logging.info("="*50)
            logging.info("\nKL Divergence Ranking (lower is better):")
            for i, (name, kl_div) in enumerate(kl_ranking[:10]):
                logging.info(f"{i+1:2d}. {name:15s}: {kl_div:.6f}")
            logging.info("\nAIC Ranking (lower is better):")
            for i, (name, aic) in enumerate(aic_ranking[:10]):
                logging.info(f"{i+1:2d}. {name:15s}: {aic:.2f}")
            logging.info("\nBIC Ranking (lower is better):")
            for i, (name, bic) in enumerate(bic_ranking[:10]):
                logging.info(f"{i+1:2d}. {name:15s}: {bic:.2f}")
            logging.info("\nKolmogorov-Smirnov Test Ranking (lower statistic is better):")
            for i, (name, ks_stat) in enumerate(ks_ranking[:10]):
                ks_p = results[name]['ks_pvalue']
                logging.info(f"{i+1:2d}. {name:15s}: KS={ks_stat:.4f}, p-value={ks_p:.4f}")

            # فراخوانی تابع برای تعیین توزیع‌های برتر کلی
            criteria_metrics_list = ['kl_divergence', 'aic', 'bic', 'ks_statistic']
            criteria_types_list = ['cost', 'cost', 'cost', 'cost']
            user_custom_weights = self.ahp_weights(criteria_metrics_list)
            logging.info("\nفراخوانی روش TOPSIS برای رتبه‌بندی کلی...")
            overall_top_5_topsis = self.topsis_ranking(results, criteria_metrics_list, criteria_types_list, criteria_weights_dict=user_custom_weights, num_top=5)
            if overall_top_5_topsis:
                logging.info(f"\nلیست نام {len(overall_top_5_topsis)} توزیع برتر کلی بر اساس TOPSIS: {overall_top_5_topsis}")
            else:
                logging.warning("TOPSIS نتوانست توزیع برتری را مشخص کند.")

            # رسم نمودار مقایسه بهترین توزیع‌ها
            plt.figure(figsize=(15, 10))
            plt.hist(mean_predictions_unscaled, bins=50, density=True, alpha=0.7, label='Empirical Data', color='lightblue', edgecolor='black')
            plt.plot(x, y_true, 'k-', linewidth=2, label='KDE (Reference)')
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for i, name in enumerate(overall_top_5_topsis):
                if name in results and results[name].get('y_pred') is not None:
                    plt.plot(x, results[name]['y_pred'], '--', color=colors[i % len(colors)], linewidth=2, label=f'{name} (Overall Top {i+1})')
            plt.xlabel('Value', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.title('Distribution Comparison - Top 5 by Combined Ranking', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{output_dir}/improved_distribution_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

            # ذخیره نتایج در فایل
            results_df = pd.DataFrame([
                {
                    'Distribution': name,
                    'KL_Divergence': res['kl_divergence'],
                    'AIC': res['aic'],
                    'BIC': res['bic'],
                    'LogLikelihood': res['loglikelihood'],
                    'N_Params': res['n_params'],
                    'KS_Statistic': res['ks_statistic'],
                    'KS_PValue': res['ks_pvalue'],
                    'Parameters': str(res['params'])
                }
                for name, res in results.items()
            ])
            results_df = results_df.sort_values('KL_Divergence')
            results_df.to_excel(f"{output_dir}/improved_distribution_comparison_results.xlsx", index=False)
            logging.info(f"\nDetailed results saved to: {output_dir}/improved_distribution_comparison_results.xlsx")
            logging.info(f"Comparison plot saved to: {output_dir}/improved_distribution_comparison.png")

            # ذخیره 5 توزیع برتر در یک فایل اکسل جداگانه
            top_5_df = results_df[results_df['Distribution'].isin(overall_top_5_topsis)]
            top_5_df.to_excel(f"{output_dir}/top_5_distributions.xlsx", index=False)
            logging.info(f"Top 5 distributions saved to: {output_dir}/top_5_distributions.xlsx")
            return results_df
        else:
            logging.error("No valid distributions found for comparison")
            return None

    def main(self):
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
        output_dir = r"C:\Users\masoud\results\P\sqp_transformer_final\final\new"
        os.makedirs(output_dir, exist_ok=True)
        try:
            final_model = SplineQuantileRegressionModel(config=best_config)
            X_train, y_train = final_model.load_data(train_features, train_target)
            X_test, y_test = final_model.load_data(test_features, test_target)
            X_val, X_final_test, y_val, y_final_test = train_test_split(X_test, y_test, test_size=0.5, random_state=SEED, shuffle=False)
            logging.info(f"Data split: Validation ({X_val.shape[0]} samples), Final Test ({X_final_test.shape[0]} samples).")
            naive_error_train = np.mean(np.abs(y_train[1:] - y_train[:-1])) if len(y_train) > 1 else np.nan
            logging.info(f"Naive error for MASE: {naive_error_train}")
            checkpoint_dir = os.path.join(output_dir, "checkpoint")
            train_loss, val_loss = final_model.train_model(X_train, y_train, validation_data=(X_val, y_val), checkpoint_dir=checkpoint_dir)
            logging.info(f"Training completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            final_model.train_calibrator(X_val, y_val, num_samples_calibration=200000, save_dir=output_dir)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            predictions_val, valid_indices_val = final_model.predict_distribution_calibrated(X_val, num_samples=best_config['num_samples'])
            y_val_valid = y_val[valid_indices_val]
            final_model.plot_results(y_val_valid, predictions_val, title='Validation Set', save_dir=output_dir, scaler=scaler)
            predictions_test, valid_indices_test = final_model.predict_distribution_calibrated(X_final_test, num_samples=best_config['num_samples'])
            y_test_valid = y_final_test[valid_indices_test]
            final_model.plot_results(y_test_valid, predictions_test, title='Test Set', save_dir=output_dir, scaler=scaler)
            plt.figure(figsize=(12, 12))
            plt.subplot(2, 1, 1)
            img_val = plt.imread(os.path.join(output_dir, 'Validation Set_predicted_distribution.png'))
            plt.imshow(img_val)
            plt.axis('off')
            plt.subplot(2, 1, 2)
            img_test = plt.imread(os.path.join(output_dir, 'Test Set_predicted_distribution.png'))
            plt.imshow(img_test)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'combined_predicted_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
            mean_predictions_unscaled = scaler.inverse_transform(np.mean(predictions_test, axis=1).reshape(-1, 1)).flatten()
            logging.info(f"Data distribution stats: mean={np.mean(mean_predictions_unscaled):.4f}, std={np.std(mean_predictions_unscaled):.4f}, min={np.min(mean_predictions_unscaled):.4f}, max={np.max(mean_predictions_unscaled):.4f}")
            confidence_levels = [0.90, 0.95, 0.99]
            risk_metrics = []
            for conf in confidence_levels:
                alpha = 1 - conf
                beta = alpha / 2
                omega1 = conf / 2
                omega2 = conf / 2
                var_neg = final_model.calculate_var(mean_predictions_unscaled, alpha, is_left_tail=True)
                cvar_neg = final_model.calculate_cvar(mean_predictions_unscaled, alpha, is_left_tail=True)
                gluevar_neg = final_model.calculate_glue_var(mean_predictions_unscaled, alpha, beta, omega1, omega2, is_left_tail=True)
                risk_metrics.append({
                    'Confidence Level': conf,
                    'VaR_Negative': var_neg,
                    'CVaR_Negative': cvar_neg,
                    'GlueVaR_Negative': gluevar_neg
                })
                logging.info(f"Confidence {conf*100:.0f}%: VaR_Neg={var_neg:.4f}, CVaR_Neg={cvar_neg:.4f}, GlueVaR_Neg={gluevar_neg:.4f}, omega1={omega1:.4f}, omega2={omega2:.4f}")
            confidence_levels_sets = [[0.01, 0.99], [0.05, 0.95], [0.10, 0.90]]
            plt.figure(figsize=(12, 18))
            for i, conf_levels in enumerate(confidence_levels_sets, 1):
                plt.subplot(3, 1, i)
                sns.kdeplot(mean_predictions_unscaled, label='Predicted Distribution', color='black', fill=False, linewidth=2)
                for conf in conf_levels:
                    conf_idx = next((j for j, d in enumerate(risk_metrics) if d['Confidence Level'] == (1 - conf)), None)
                    if conf_idx is not None:
                        metrics = risk_metrics[conf_idx]
                        if conf < 0.5:
                            var_value = metrics['VaR_Negative']
                            cvar_value = metrics['CVaR_Negative']
                            glue_var_value = metrics['GlueVaR_Negative']
                            plt.axvline(x=var_value, color='red', linestyle='--', label=f'VaR {conf:.2f}')
                            plt.axvline(x=cvar_value, color='blue', linestyle='--', label=f'CVaR {conf:.2f}')
                            plt.axvline(x=glue_var_value, color='green', linestyle='--', label=f'GlueVaR {conf:.2f}')
                plt.xlabel('Value (Original Scale)', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'test_set_kde_with_risk_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            df_risk_metrics = pd.DataFrame(risk_metrics)
            df_risk_metrics.to_excel(os.path.join(output_dir, 'risk_metrics.xlsx'), index=False)
            logging.info(f"Risk metrics saved to: {output_dir}/risk_metrics.xlsx")
            logging.info("Starting comparison with other distributions")
            results_df = final_model.improved_distribution_comparison(mean_predictions_unscaled, output_dir)
            final_model.model.save(os.path.join(output_dir, 'sqp_transformer_model_final.keras'))
            logging.info(f"Model saved to: {output_dir}/sqp_transformer_model_final.keras")
        except Exception as e:
            logging.error(f"Error in main: {str(e)}")
            raise

if __name__ == "__main__":
    model = SplineQuantileRegressionModel(best_config)
    model.main()
