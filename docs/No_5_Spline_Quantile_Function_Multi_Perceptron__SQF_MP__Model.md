# No.5  - Spline Quantile Function Multi Perceptron (SQF_MP) Model.py

This code implements an advanced framework for Spline-based Quantile Regression applied to financial time series data, particularly Bitcoin returns. It integrates machine learning, statistical modeling and distributed hyperparameter optimization to identify the most suitable probability distribution and quantify risk metrics.

Key Components

1. Data Handling and Preprocessing:
   - Loads raw time series data and prepares training/testing datasets.
   - Applies transformations such as signed logarithm, scaling, and outlier detection.
   - Splits data into train/test subsets for robust evaluation.

2. Spline Quantile Regression Model:
   - Defines a spline-enhanced neural network for quantile regression using TensorFlow/Karas.
   - Employs custom quantile loss (pinball loss) to estimate conditional quantiles.
   - Flexible enough to capture non-linear patterns in heavy-tailed financial returns.

3. Hyperparameter Optimization:
   - Uses **Ray Tune** and **HyperOpt** for distributed hyperparameter search.
   - Bayesian and early-stopping strategies ensure efficient convergence.
   - Evaluates candidate models with K-fold cross-validation.

4. Statistical and Risk Metrics:
   - Computes skewness, kurtosis, eBIC (Extended Bayesian Information Criterion), and other measures.
   - Evaluates coverage probabilities, quantile accuracy, and predictive stability.
   - Focused on financial risk applications such as Value-at-Risk (VaR).

5. Feature Engineering and Selection:
   - Incorporates feature preprocessing, leakage detection, and correlation filtering.
   - Uses XGBoost and SHAP analysis to rank feature importance.
   - Produces correlation heatmaps and feature importance plots for interpretability.

6. Visualization and Reporting:
   - Generates diagnostic plots including histograms, quantile fits, and actual vs. predicted returns.
   - Produces SHAP summary and importance plots for model interpretability.
   - Saves outputs (CSV, Excel, PKL, PNG) for reproducibility and reviewer access.

7. Reproducibility and Logging:
   - All experiments are logged with structured outputs and random seeds for determinism.
   - Previous results are automatically cleared to avoid contamination.
   - Intermediate artifacts (models, results, feature lists) are stored systematically.

Workflow
1. Load and preprocess financial time series data.
2. Select features using Spearman correlation, XGBoost filtering, and SHAP analysis.
3. Define and train a spline quantile regression model.
4. Optimize hyperparameters via distributed search.
5. Evaluate performance on test data using risk and distribution metrics.
6. Save results, generate plots, and create reproducible artifacts.

Applications
- Risk Management: Estimation of Value-at-Risk (VaR) and Expected Shortfall (ES).
- Probability Modeling: Capturing fat-tailed distributions of cryptocurrency returns.
- Financial Forecasting: Nonlinear prediction under uncertainty.

 Output
The script produces:
- Trained models and hyperparameter configurations.
- Feature importance rankings and SHAP-based interpretability results.
- Statistical summaries and risk metrics.
- Visual plots for distribution analysis and prediction diagnostics.

This comprehensive framework balances predictive accuracy, interpretability, and reproducibility, making it suitable for both academic research and practical financial applications.
