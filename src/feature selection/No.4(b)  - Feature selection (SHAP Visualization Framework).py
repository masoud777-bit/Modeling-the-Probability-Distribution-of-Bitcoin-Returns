# -*- coding: utf-8 -*-
"""
SHAP Visualization Framework for Feature Importance Analysis
------------------------------------------------------------

This script generates enhanced SHAP summary and importance plots
based on a trained XGBoost model and selected financial features.
It supports both loading pre-trained models and reconstructing models
from stored training data and optimal hyperparameters.

Author: [M.fadakar]
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBRegressor


# --------------------------------------------------------
# Global Configurations
# --------------------------------------------------------
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 8,
    'legend.fontsize': 10,
    'font.family': 'DejaVu Sans'
})

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = r'C:\Users\masoud\feature_selection_results'
VERSION = 'v8_separate_train_test'


def get_path(filename: str) -> str:
    """Generate versioned file path inside BASE_DIR."""
    return os.path.join(BASE_DIR, f'{filename}_{VERSION}')


# Input files
DATA_STORAGE_FILE = get_path('data_storage_train') + '.pkl'
FINAL_MODEL_FILE = get_path('final_model_train') + '.pkl'
SHAP_IMPORTANCE_FILE = get_path('shap_importance_train') + '.csv'

# Output files
NEW_SHAP_SUMMARY_PLOT_FILE = get_path('shap_summary_plot_train_updated') + '.png'
NEW_SHAP_IMPORTANCE_PLOT_FILE = get_path('shap_importance_plot_train_updated') + '.png'


# --------------------------------------------------------
# Data and Model Loading
# --------------------------------------------------------
def load_data_and_model():
    """Load training data and final model from pickle files."""
    try:
        with open(DATA_STORAGE_FILE, 'rb') as f:
            data_storage = pickle.load(f)

        X_train_final = data_storage['X_train_final']
        y_train = data_storage['y_train']
        final_selected_features = data_storage['final_selected_features']
        logging.info(f"Training data loaded: {X_train_final.shape}")

        model_file_path = get_path('final_model_train')
        with open(model_file_path, 'rb') as f:
            final_model = pickle.load(f)
        logging.info("Final model loaded successfully")

        return X_train_final, y_train, final_selected_features, final_model

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return load_data_and_recreate_model()
    except Exception as e:
        logging.error(f"Error loading data/model: {e}")
        return None, None, None, None


def load_data_and_recreate_model():
    """Recreate model if the saved model file is missing."""
    try:
        with open(DATA_STORAGE_FILE, 'rb') as f:
            data_storage = pickle.load(f)

        X_train_final = data_storage['X_train_final']
        y_train = data_storage['y_train']
        final_selected_features = data_storage['final_selected_features']

        bayes_opt_file = get_path('bayes_opt_results_train') + '.pkl'
        with open(bayes_opt_file, 'rb') as f:
            bayes_results = pickle.load(f)
            best_params = bayes_results['best_params']

        final_model = XGBRegressor(random_state=42, **best_params)
        final_model.fit(X_train_final, y_train)
        logging.info("Model successfully reconstructed")

        return X_train_final, y_train, final_selected_features, final_model

    except Exception as e:
        logging.error(f"Error reconstructing model: {e}")
        return None, None, None, None


# --------------------------------------------------------
# SHAP Visualization Functions
# --------------------------------------------------------
def create_updated_shap_summary_plot(final_model, X_train_final, top_percent=30):
    """Generate SHAP summary plot for top percentage of features."""
    try:
        logging.info("Generating SHAP summary plot...")
        explainer = shap.TreeExplainer(final_model)

        sample_size = min(100, len(X_train_final))
        X_sample = X_train_final.sample(n=sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)

        feature_importance = np.abs(shap_values).mean(axis=0)
        n_features = int(len(feature_importance) * top_percent / 100)
        top_indices = np.argsort(feature_importance)[-n_features:]

        X_sample_top = X_sample.iloc[:, top_indices]
        shap_values_top = shap_values[:, top_indices]

        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values_top,
            X_sample_top,
            plot_type="dot",
            show=False,
            max_display=n_features
        )
        plt.title(f'SHAP Summary Plot - Top {top_percent}% Features',
                  fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(NEW_SHAP_SUMMARY_PLOT_FILE, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Summary plot saved: {NEW_SHAP_SUMMARY_PLOT_FILE}")

    except Exception as e:
        logging.error(f"Error creating SHAP summary plot: {e}")


def create_updated_shap_importance_plot(final_model, X_train_final, top_percent=70):
    """Generate SHAP importance barplot for top percentage of features."""
    try:
        logging.info("Generating SHAP importance plot...")
        explainer = shap.TreeExplainer(final_model)

        sample_size = min(100, len(X_train_final))
        X_sample = X_train_final.sample(n=sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)

        shap_importance = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=X_train_final.columns
        ).sort_values(ascending=False)

        n_features = int(len(shap_importance) * top_percent / 100)
        top_features = shap_importance.head(n_features)

        shap_percentage = (top_features / shap_importance.sum()) * 100
        plot_data = pd.DataFrame({
            'Feature': shap_percentage.index,
            'SHAP Importance (%)': shap_percentage.values
        })

        plt.figure(figsize=(14, max(8, len(plot_data) * 0.3)))
        ax = sns.barplot(
            data=plot_data,
            x='SHAP Importance (%)',
            y='Feature',
            palette='viridis',
            orient='h'
        )
        ax.set_yticklabels(plot_data['Feature'], rotation=0,
                           fontsize=8, ha='right')
        ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        plt.title(f'SHAP Feature Importance - Top {top_percent}% Features',
                  fontsize=14, pad=20)
        plt.xlabel('SHAP Importance (%)', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(NEW_SHAP_IMPORTANCE_PLOT_FILE,
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        csv_file = get_path('shap_importance_top70_train_updated') + '.csv'
        plot_data.to_csv(csv_file, index=False, encoding='utf-8')
        logging.info(f"Importance plot and CSV saved: {csv_file}")

    except Exception as e:
        logging.error(f"Error creating SHAP importance plot: {e}")


# --------------------------------------------------------
# Main Execution
# --------------------------------------------------------
def main():
    """Main execution function for SHAP visualization."""
    logging.info("Starting SHAP visualization pipeline...")

    if not os.path.exists(BASE_DIR):
        logging.error(f"Directory {BASE_DIR} does not exist!")
        return

    logging.info("Available files in BASE_DIR:")
    for file in os.listdir(BASE_DIR):
        if file.endswith(('.pkl', '.csv')):
            logging.info(f"  - {file}")

    X_train_final, y_train, final_selected_features, final_model = load_data_and_model()
    if any(x is None for x in [X_train_final, y_train, final_selected_features, final_model]):
        logging.error("Failed to load data/model. Process stopped.")
        return

    create_updated_shap_summary_plot(final_model, X_train_final, top_percent=30)
    create_updated_shap_importance_plot(final_model, X_train_final, top_percent=70)

    logging.info("SHAP visualization completed successfully!")

    print("\nGenerated files:")
    print(f"1. SHAP Summary Plot (Top 30% features): {NEW_SHAP_SUMMARY_PLOT_FILE}")
    print(f"2. SHAP Importance Plot (Top 70% features): {NEW_SHAP_IMPORTANCE_PLOT_FILE}")


if __name__ == "__main__":
    main()
