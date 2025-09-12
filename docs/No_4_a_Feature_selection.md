# No.4(a)  - Feature selection.py


This script contains the implementation of a comprehensive framework for feature selection and predictive modeling in financial time series analysis, with a focus on Bitcoin risk evaluation.
The project integrates multiple advanced machine learning techniques to identify the most informative predictors, reduce redundancy, and build interpretable models. It is designed to ensure reproducibility, robustness, and transparency for academic research and peer review.

Key Components:
---------------
1. Data Preprocessing and Validation:
   - Loading raw financial and blockchain-related datasets.
   - Automatic detection and handling of missing, invalid, or 
     inconsistent values.
   - Separation of training and testing datasets to prevent 
     information leakage.

2. Feature Selection Pipeline:
   - Spearman correlation filtering to remove irrelevant features.
   - XGBoost-based importance ranking to refine feature subsets.
   - Correlation heatmaps for visual analysis of multicollinearity.

3. Model Training and Optimization:
   - Random Forest and XGBoost regressors as base models.
   - Hyperparameter tuning via Bayesian Optimization 
     (BayesSearchCV with early stopping).
   - K-fold cross-validation for robust performance estimation.

4. Model Evaluation:
   - Multiple performance metrics including RMSE, MAE, R², 
     and percentage-based errors (MAPE, MdAPE).
   - Comparison of actual vs. predicted values on unseen test data.

5. Interpretability and Transparency:
   - SHAP (SHapley Additive exPlanations) for feature importance 
     and interpretability of predictions.
   - Leakage analysis to ensure selected features do not 
     introduce hidden biases.

6. Output and Reproducibility:
   - Automated saving of selected features, evaluation results, 
     SHAP values, correlation heatmaps, and model configurations.
   - All intermediate results (e.g., filtered feature lists, 
     hyperparameter search results) are logged and stored.

Use Case:
---------
The framework was applied to Bitcoin data (economic indicators, technical features, and blockchain-related metrics) to identify key risk drivers and build a probability distribution model. 
However, the codebase is general and can be adapted for other financial assets or domains requiring rigorous feature selection and model evaluation.

Target Audience:
----------------
- Academic researchers in finance, econometrics, and machine learning.
- Practitioners seeking transparent and reproducible risk modeling methods.
- Reviewers and peers evaluating research reproducibility.