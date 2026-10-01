# No.9 - Probabilistic Forecasting by SQF_MT.py

Spline Quantile Function Multi Transformer (SQF-MT) is an advanced deep learning model designed for high-resolution probabilistic time series forecasting. It integrates powerful. 
Transformer architecture with a non-parametric Spline-Based Quantile Regression head. Instead of predicting a single point estimate, this model forecasts a complete probability distribution for each time step, enabling robust uncertainty quantification.
This implementation is a full-fledged pipeline for financial risk management. The core methodology involves:
Processing sequential data using a multi-layer Transformer encoder to learn temporal patterns.
Predicting the coefficients of a natural cubic spline.
Interpolating these coefficients to construct a smooth, continuous quantile function (the inverse CDF).
Generating a full predictive distribution from this quantile function.

Key Features
Core Architecture:
Transformer Encoder: Utilizes multi-head self-attention mechanisms with positional encoding to effectively capture long-range dependencies in sequential data.
Spline Quantile Regression: Implements B-spline interpolation across a high-density set of quantiles. This approach predicts a small number of spline knot coefficients, which are then used to generate a smooth and continuous quantile function, offering a more stable and efficient alternative to predicting hundreds of quantiles directly.
Probabilistic Calibration: Includes a post-processing calibration step using Isotonic Regression. This non-parametric method adjusts the model's output by analyzing its Probability Integral Transform (PIT) values on a validation set, significantly improving the reliability and accuracy of the prediction intervals.



Advanced Capabilities:

Advanced Risk Management Module: Features built-in functions to calculate critical financial risk metrics directly from the predicted distributions, including Value-at-Risk (VaR) , Conditional Value-at-Risk (CVaR) , and the more sophisticated GlueVaR.
Multi-Criteria Distribution Analysis: Includes a comprehensive framework (improved_distribution_comparison) to benchmark the model's predicted distribution against 20 standard statistical distributions (e.g., Johnson SU, Normal, Log-Normal). The best-fitting distributions are identified using a multi-criteria decision-making process based on the 
Analytic Hierarchy Process (AHP) for weighting and the TOPSIS method for ranking. This process considers metrics like KL Divergence, AIC, BIC, and the KS-statistic.
Comprehensive Visualization Suite: Generates a rich set of plots for in-depth analysis, including time series forecasts with prediction intervals , PIT histograms for calibration assessment , training and validation loss curves , and density comparisons against best-fit distributions.
Reproducibility: Enforces reproducibility by setting a fixed seed across all relevant libraries (NumPy, TensorFlow, random), ensuring that experiments can be repeated with identical results.

Technical Implementation
Model Architecture
The data flows through the model as follows:
Input Sequence → Positional Encoding → N x Transformer Encoder Layers → Dense Layer (predicts spline coefficients).
The final layer outputs a vector of coefficients corresponding to the spline knots, which are then used to construct the full quantile function.

Quantile Selection Strategy
To capture tail risk accurately without excessive computational cost, the model uses an adaptive quantile selection strategy:
Dense Tails: The 1%-10% and 90%-100% quantile ranges are sampled densely.
Sparse Middle: The central 10%-90% range is sampled more sparsely.
This strategy results in a high-resolution view of the entire distribution with a focus on the tails.

Loss Function
The model is trained by minimizing the Quantile Loss (also known as the Pinball Loss) averaged across all defined quantiles. The loss for a single quantile τ is given by:
L(y, q) = Σ max(τ(y - q_τ), (τ - 1)(y - q_τ)) 

where :
y is the true value and q_τ is the predicted value at quantile τ.
Execution Workflow The main() function orchestrates the end-to-end workflow:

Data Loading: Loads training and testing features and targets from specified .xlsx files.
Data Splitting: Splits the test set into validation and final test sets using a 50/50 split.
Model Training: Initializes and trains the SplineQuantileRegressionModel using EarlyStopping and ReduceLROnPlateau for efficient training. Model checkpoints are saved to resume training or restore the best model.
Model Calibration: Trains the isotonic regression calibrator on the validation set predictions.
Prediction & Plotting: Generates calibrated distributional forecasts for both validation and test sets and saves all relevant plots.
Risk Analysis: Calculates VaR, CVaR, and GlueVaR at 90%, 95%, and 99% confidence levels and saves them to an Excel file.
Distribution Comparison: Runs the improved_distribution_comparison analysis to find the best-fitting standard distributions and saves the detailed results and summary plots.
Model Saving: Saves the final trained and calibrated model to a .keras file.
Output Analysis

The script generates a comprehensive set of artifacts in the specified output directory:
Plots (.png files):
raw_pit_vs_ranks_plot.png: Q-Q plot of raw PIT values for calibration assessment.
{Validation/Test} Set_predicted_distribution.png: Time series plot of actual values, predicted mean, and the 95% prediction interval.
{Validation/Test} Set_training_history.png: Training and validation loss curves over epochs.
test_set_kde_with_risk_metrics.png: Density plot of the test set predictions, annotated with VaR, CVaR, and GlueVaR lines.
improved_distribution_comparison.png: A plot comparing the empirical data histogram, the model's KDE, and the PDFs of the top 5 best-fitting standard distributions.

Data Files (.xlsx files):
risk_metrics.xlsx: A table of calculated VaR, CVaR, and GlueVaR values at different confidence levels.
improved_distribution_comparison_results.xlsx: Detailed results of the distribution fitting analysis, including all metrics for all 20 distributions.
top_5_distributions.xlsx: A filtered list containing only the top 5 best-fitting distributions as determined by the TOPSIS ranking.

Model File:
sqp_transformer_model_final.keras: The final, saved TensorFlow/Keras model, ready for inference.
Of course. Here is a comprehensive "Extended Description (English)" for the provided Python code, suitable for a project's documentation or a GitHub README.
Spline Quantile Transformer (SQF-MT) Model for Probabilistic Forecasting
Overview
The Spline Quantile Function Multi Transformer (SQF-MT) is an advanced deep learning model designed for high-resolution probabilistic time series forecasting. It integrates a powerful Trasformer architecture with a non-parametric Spline-Based Quantile Regression head. Instead of predicting a single point estimate, this model forecasts a complete probability distribution for each time step, enabling robust uncertainty quantification.




This implementation is a full-fledged pipeline for financial risk management. The core methodology involves:
Processing sequential data using a multi-layer Transformer encoder to learn temporal patterns.
Predicting the coefficients of a natural cubic spline.
Interpolating these coefficients to construct a smooth, continuous quantile function (the inverse CDF).
Generating a full predictive distribution from this quantile function.

Key Features
Core Architecture:
Transformer Encoder: Utilizes multi-head self-attention mechanisms with positional encoding to effectively capture long-range dependencies in sequential data.
Spline Quantile Regression: Implements B-spline interpolation across a high-density set of quantiles. This approach predicts a small number of spline knot coefficients, which are then used to generate a smooth and continuous quantile function, offering a more stable and efficient alternative to predicting hundreds of quantiles directly.
Probabilistic Calibration: Includes a post-processing calibration step using Isotonic Regression. This non-parametric method adjusts the model's output by analyzing its Probability Integral Transform (PIT) values on a validation set, significantly improving the reliability and accuracy of the prediction intervals.



Advanced Capabilities:

Advanced Risk Management Module: Features built-in functions to calculate critical financial risk metrics directly from the predicted distributions, including Value-at-Risk (VaR) , 
Conditional Value-at-Risk (CVaR) , and the more sophisticated GlueVaR.
Multi-Criteria Distribution Analysis: Includes a comprehensive framework (improved_distribution_comparison) to benchmark the model's predicted distribution against 20 standard statistical distributions (e.g., Johnson SU, Normal, Log-Normal). The best-fitting distributions are identified using a multi-criteria decision-making process based on the 
Analytic Hierarchy Process (AHP) for weighting and the TOPSIS method for ranking. This process considers metrics like KL Divergence, AIC, BIC, and the KS-statistic.
Comprehensive Visualization Suite: Generates a rich set of plots for in-depth analysis, including time series forecasts with prediction intervals , PIT histograms for calibration assessment , training and validation loss curves , and density comparisons against best-fit distributions.

Reproducibility: Enforces reproducibility by setting a fixed seed across all relevant libraries (NumPy, TensorFlow, random), ensuring that experiments can be repeated with identical results.

Technical Implementation
Model Architecture
The data flows through the model as follows:
Input Sequence → Positional Encoding → N x Transformer Encoder Layers → Dense Layer (predicts spline coefficients).
The final layer outputs a vector of coefficients corresponding to the spline knots, which are then used to construct the full quantile function.

Quantile Selection Strategy
To capture tail risk accurately without excessive computational cost, the model uses an adaptive quantile selection strategy:
Dense Tails: The 1%-10% and 90%-100% quantile ranges are sampled densely.
Sparse Middle: The central 10%-90% range is sampled more sparsely.
This strategy results in a high-resolution view of the entire distribution with a focus on the tails.

Loss Function
The model is trained by minimizing the 
Quantile Loss (also known as the Pinball Loss) averaged across all defined quantiles. The loss for a single quantile 
τ is given by:
L(y, q) = Σ max(τ(y - q_τ), (τ - 1)(y - q_τ)) 
where 
y is the true value and q_τ is the predicted value at quantile τ.

Execution Workflow
The main() function orchestrates the end-to-end workflow:
Data Loading: Loads training and testing features and targets from specified .xlsx files.
Data Splitting: Splits the test set into validation and final test sets using a 50/50 split.
Model Training: Initializes and trains the SplineQuantileRegressionModel using EarlyStopping and ReduceLROnPlateau for efficient training. Model checkpoints are saved to resume training or restore the best model.
Model Calibration: Trains the isotonic regression calibrator on the validation set predictions.
Prediction & Plotting: Generates calibrated distributional forecasts for both validation and test sets and saves all relevant plots.
Risk Analysis: Calculates VaR, CVaR, and GlueVaR at 90%, 95%, and 99% confidence levels and saves them to an Excel file.
Distribution Comparison: Runs the improved_distribution_comparison analysis to find the best-fitting standard distributions and saves the detailed results and summary plots.
Model Saving: Saves the final trained and calibrated model to a .keras file.

Output Analysis
The script generates a comprehensive set of artifacts in the specified output directory:

Plots (.png files):
raw_pit_vs_ranks_plot.png: Q-Q plot of raw PIT values for calibration assessment.
{Validation/Test} Set_predicted_distribution.png: Time series plot of actual values, predicted mean, and the 95% prediction interval.
{Validation/Test} Set_training_history.png: Training and validation loss curves over epochs.
test_set_kde_with_risk_metrics.png: Density plot of the test set predictions, annotated with VaR, CVaR, and GlueVaR lines.
improved_distribution_comparison.png: A plot comparing the empirical data histogram, the model's KDE, and the PDFs of the top 5 best-fitting standard distributions.

Data Files (.xlsx files):
risk_metrics.xlsx: A table of calculated VaR, CVaR, and GlueVaR values at different confidence levels.
improved_distribution_comparison_results.xlsx: Detailed results of the distribution fitting analysis, including all metrics for all 20 distributions.
top_5_distributions.xlsx: A filtered list containing only the top 5 best-fitting distributions as determined by the TOPSIS ranking.

Model File:
sqp_transformer_model_final.keras: The final, saved TensorFlow/Keras model, ready for inference.
Dependencies
Core:
tensorflow / keras 
numpy 
pandas 
scikit-learn 

Visualization:
matplotlib 
seaborn 
Statistical Analysis:
scipy 
properscoring 

File I/O:
xlsxwriter 
pickle