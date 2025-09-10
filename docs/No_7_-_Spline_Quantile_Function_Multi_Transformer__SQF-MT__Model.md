# No.7 - Spline Quantile Function Multi Transformer (SQF-MT) Model.py

Spline-Based Quantile Regression with a Transformer Encoder.

This script implements a complete machine learning pipeline for time-series
forecasting using a transformer-based model for spline quantile regression.
The pipeline includes data loading, preprocessing, hyperparameter tuning with
Ray Tune, model training, probabilistic calibration, and comprehensive evaluation.

Key components:
- SplineQuantileRegressionModel: A class that encapsulates the entire model,
  including the transformer architecture, spline-based quantile prediction,
  and calibration logic.
- tune_model: A function that defines the training and evaluation process for
  a single trial in the Ray Tune hyperparameter search.
- main: The main function that orchestrates the entire workflow, from setting
  up the environment to running the hyperparameter search and evaluating the
  final model.

Author: [M.fadakar]
