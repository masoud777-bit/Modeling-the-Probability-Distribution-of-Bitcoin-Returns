# No.3 - preprocessing of features.py

This script performs advanced preprocessing of financial time series data by applying a signed log transformation (signed_log1p), detecting and replacing outliers through quantile regression, and normalizing the values using a RobustScaler.
Constant or all-NaN columns are identified and handled separately to prevent distortions. The data is split into training and test subsets, transformed, and finally saved into Excel files for further analysis. Descriptive statistics (skewness and kurtosis) and distribution plots are generated for selected variables to provide deeper insights into the processed dataset.
