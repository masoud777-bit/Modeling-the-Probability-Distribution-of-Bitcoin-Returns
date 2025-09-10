# No.8 - Spline Quantile Function Multi LSTM(SQF_MLSTM) Model.py

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
