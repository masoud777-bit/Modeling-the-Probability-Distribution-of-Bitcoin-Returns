# No.8 - Spline Quantile Function Multi LSTM(SQF_MLSTM) Model.py

This repository presents a sophisticated and comprehensive solution for probabilistic time series forecasting using a Spline Quantile Regression model implemented with Long Short-Term Memory (LSTM) networks. Unlike traditional forecasting methods that provide a single point prediction, this approach estimates the full predictive distribution, which is essential for risk analysis and decision-making under uncertainty.

Key Features
Probabilistic Forecasting: The model goes beyond point estimates to predict a complete quantile distribution, offering a detailed view of forecast uncertainty. This is achieved by estimating multiple quantiles simultaneously, which together define the shape of the predictive distribution.
Spline-Based Quantile Regression: A key innovation of this model is the use of cubic splines to parameterize the quantile function. This non-parametric approach provides exceptional flexibility, allowing the model to capture complex, non-linear dependencies in the data without making rigid assumptions about the distribution's shape.
LSTM Architecture: The core of the forecasting engine is a deep LSTM neural network. LSTMs are uniquely suited for sequential data, as they can learn and remember long-term temporal dependencies, making them highly effective for time series analysis.
Automated Hyperparameter Tuning: The project incorporates Ray Tune for efficient and automated hyperparameter optimization. This allows the model to intelligently search for the best configuration (e.g., number of LSTM units, layers, and learning rate) to maximize its predictive performance.
Model Calibration: A crucial step in ensuring the reliability of probabilistic forecasts is calibration. This codebase includes an isotonic regression-based calibrator to correct any systematic biases in the predictive distributions, ensuring that the predicted quantiles are statistically sound and that the model is well-calibrated.
Comprehensive Evaluation: The model's performance is rigorously evaluated using a suite of advanced metrics, including CRPS (Continuous Ranked Probability Score), RMSE, MAE, R², and Wasserstein distance. In addition, the project provides a rich set of visualizations to help understand the model's predictions, training history, and distribution quality.

Technologies and Libraries
The project is built on a modern Python stack, leveraging the following key libraries:
TensorFlow/Keras: For building and training the deep learning model.
Ray/Ray Tune: For distributed computing and hyperparameter optimization.
Pandas & NumPy: For efficient data handling and numerical operations.
Scikit-learn: For data splitting and performance metrics.
Matplotlib & Seaborn: For creating insightful visualizations.
properscoring: For calculating the Continuous Ranked Probability Score (CRPS).

Getting Started
The code is designed to be highly modular and easy to use. To run this project, you will need to:
Prepare Your Data: Ensure your time series data is in a compatible format. The provided code example uses .xlsx files with a target variable column named 'lreturn'.
Initialize the Model: Create an instance of the SplineQuantileRegressionModel class with a configuration dictionary defining the model's structure.
Train the Model: Use the train_model method to fit the model to your training data.
Calibrate Predictions: The train_calibrator method can be used to improve the statistical validity of the quantile predictions.
Generate Predictions: The predict_distribution_calibrated method provides calibrated forecast samples for a given input.
Evaluate and Visualize: Utilize the calculate_metrics and plot_results methods to assess performance and generate plots.

Project Structure
The main script (main()) orchestrates the entire workflow, from data loading to hyperparameter tuning via Ray Tune and final model evaluation. Key functions include tune_model(), which wraps the hyperparameter search, and SplineQuantileRegressionModel, which contains the core model logic. The script also includes comprehensive logging to track the progress of each step.