# No.6 - Spline Quantile Function Multi GRU (SQF_MGRU) Model.py

This script implements a Spline Quantile Regression Model with GRU architecture for probabilistic time series forecasting in finance. It integrates deep learning, spline-based quantile regression, and hyperparameter optimization within a robust experimental framework designed for reproducibility and interpretability.

Key Features:

1- Model Design
A custom class SplineQuantileRegressionModel defines the core framework.
Combines Generalized Recurrent Units (GRUs) with spline-based quantile regression to capture non-linear, heavy-tailed financial return dynamics.
Employs a custom quantile loss function (pinball loss) across a large set of quantiles for accurate distributional modeling.

2- Hyperparameter Optimization
Integrates Ray Tune with HyperOpt and ASHA scheduler for distributed, scalable tuning.
Automates search across architecture (layers, units), learning rate, dropout, spline knots, and other hyperparameters.
Includes checkpointing, early stopping, and learning-rate reduction for efficient training.

3- Calibration and Reliability
Uses Probability Integral Transform (PIT) diagnostics and Isotonic Regression for distribution calibration.
Provides empirical checks (PIT histograms, CPIT plots) to validate distributional accuracy.

4- Metrics and Evaluation
Computes a wide range of evaluation metrics, including:
RMSE, MAE, MASE, R², NRMSE, sMAPE
CRPS (Continuous Ranked Probability Score)
Kullback-Leibler divergence
Wasserstein distance
Quantile Losses (QL50, QL90, QLM)
MSIS (Mean Scaled Interval Score)
Supports both scaled and original value metrics for interpretability.

5- Visualization and Outputs
Generates multiple diagnostic plots:
Training loss curves
Prediction vs. actual plots
Distribution comparison plots
Prediction error time series
PIT histograms and cumulative PIT plots
Exports trained models (.keras), scalers (.pkl), and experimental logs.

6- Workflow
Load and preprocess training/validation/test datasets.
Train the spline-quantile GRU model with calibration.
Perform hyperparameter optimization using Ray Tune.
Evaluate results with a combined objective function balancing CRPS, MSIS, R², and MAE.
Save final tuned models and plots for reproducibility.

7- Applications
Financial risk management (e.g., Value-at-Risk, Expected Shortfall).
Distributional forecasting of asset returns under heavy tails.
Stress-testing and scenario analysis for investment portfolios.
