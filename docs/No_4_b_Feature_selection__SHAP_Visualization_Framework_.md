# No.4(b)  - Feature selection (SHAP Visualization Framework).py

This script provides an automated workflow for generating improved SHAP (SHapley Additive exPlanations) visualizations to analyze feature importance in financial machinelearning models. It was developed as part of a research project on Bitcoin probability modeling and risk assessment.

Purpose:
--------
The main goal is to ensure transparency and interpretability of machine learning models by visualizing how individual features contribute to predictions. The script
supports both pre-trained models and on-demand model reconstruction based on saved training data and hyperparameters.

Key Features:
-------------
1. Data and Model Management
   - Load preprocessed training datasets and selected features.
   - Load a saved final XGBoost model or automatically rebuild it 
     using stored Bayesian optimization results.

2. SHAP Summary Plot
   - Generates a dot-based SHAP summary plot.
   - Focuses on the top 30% most important features for improved clarity.
   - Highlights distribution and magnitude of feature effects.

3. SHAP Importance Plot
   - Produces a horizontal bar plot of SHAP feature importances.
   - Visualizes the top 70% of features ranked by their mean absolute SHAP values.
   - Exports results as both PNG and CSV for reproducibility.

4. Logging and Reproducibility
   - Comprehensive logging for every step of the workflow.
   - Automatically saves all outputs (plots and CSVs) with 
     versioned file names for consistent tracking.

Application:
------------
This tool was applied to financial datasets (economic, technical, and blockchain features) for Bitcoin, but can be adapted for other assets and domains. 
It enables researchers and practitioners to better understand feature-level contributions to model predictions, supporting risk management, portfolio decisions,
and academic reproducibility.

Target Audience:
----------------
- Financial researchers applying machine learning.
- Reviewers verifying model interpretability in academic papers.
- Practitioners interested in transparent risk modeling.
