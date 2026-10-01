# No.2 - preprocessing of main feature.py

Script Name: Robust Preprocessing Pipeline for Financial Time Series
Description:
    This script implements a preprocessing pipeline for financial time series.
    Key steps include:
        - Loading and cleaning the dataset
        - Handling zeros and missing values
        - Outlier treatment with quantile regression
        - Scaling using RobustScaler (with saving and loading support)
        - Generating descriptive statistics and visualizations
        - Chronological train-test split
    Output files:
        - Preprocessed train and test sets (scaled)
        - Saved RobustScaler object
        - Optionally, inverse-transformed data
