Feature engineering on technical data.py

This Python script is a powerful tool for 'feature engineering' in quantitative finance. It is designed to transform raw OHLCV (Open, High, Low, Close, Volume) data into a rich dataset of technical indicators. By reading an input CSV file, the script automatically computes a wide array of indicators and appends them as new features, providing a robust foundation for various applications.

This process is essential for tasks such as:

 Machine Learning: Creating a comprehensive feature set for training and testing predictive models.
  Algorithmic Trading: Developing and back testing trading strategies based on technical signals.
  Quantitative Analysis: Gaining deeper insights into market behavior by analyzing derived metrics.

-----

Features

 Automated Feature Engineering: Generates over 50 different technical indicators with difference time frames, including various moving averages (SMA, EMA, WMA), momentum indicators (RSI, ROC), trend-following indicators (MACD), and volatility measures (ATR, Beta).
 Efficient and Reliable: Built on the powerful pandas, NumPy, and TA-Lib libraries, ensuring fast and accurate calculations.
 Simple Workflow: A straightforward command-line interface. Just provide your input data, and the script handles the feature generation.
Ready-to-Use Output: The output CSV contains all original data plus the newly engineered features, formatted for immediate use in your projects.

-----

How to Use:

Prerequisites

You'll need to have the required Python packages installed. The script depends on pandas, NumPy, and the TA-Lib Python wrapper.

1.  Install the TA-Lib C library first. Follow the instructions on the [TA-Lib website](https://ta-lib.org/) for your specific operating system.
2.  Install the Python packages:
    ```bash
    pip install pandas numpy TA-Lib


produced indicators base on open, high, low, close, volume:
------------------------------
ROC3, ROC7, ROC14, ROC30, ROC90,
std3, std7, std14, std30, std90,
SMA_8, SMA_13, SMA_21, SMA_34, SMA_55, SMA_100, SMA_200,
EMA_8, EMA_13, EMA_21, EMA_34, EMA_55, EMA_100, EMA_200,
WMA_8, WMA_13, WMA_21, WMA_34, WMA_55, WMA_100, WMA_200,
MACD_8, MACD_signal_8, MACD_hist_8,
MACD_13, MACD_signal_13, MACD_hist_13,
MACD_21, MACD_signal_21, MACD_hist_21,
MACD_34, MACD_signal_34, MACD_hist_34,
MACD_55, MACD_signal_55, MACD_hist_55,
RSI_5, RSI_8, RSI_13, RSI_21, RSI_34, RSI_55, RSI_89,
MOMENTUM_5, MOMENTUM_8, MOMENTUM_13, MOMENTUM_21, MOMENTUM_34, MOMENTUM_55, MOMENTUM_89,
OBV, ATR_8, ATR_13, ATR_21, ATR_34, ATR_55,
HT_TRENDLINE, AVGPRICE,
BETA_3, BETA_5, BETA_8, BETA_13, BETA_21,
WMA_volume_5, WMA_volume_8, WMA_volume_13, WMA_volume_21, WMA_volume_34, WMA_volume_55, WMA_volume_89,
AO, K_8, D_8, K_13, D_13, K_21, D_21, K_34, D_34, K_55, D_55,
DEMA_5, DEMA_8, DEMA_13, DEMA_21, DEMA_34, DEMA_55, DEMA_89

Implementation details
----------------------
- ROC is percent change: (close_t / close_{t-n} - 1) * 100
- stdN is rolling standard deviation of the close over N periods
- AO is calculated as 5-period SMA of median price minus 34-period SMA of median price    
    ```

Execution

1.  Place your OHLCV data in a CSV file named `ohlcv.csv` in the same directory as the script. Ensure your CSV has the columns: `open`, `high`, `low`, `close`, and `volume`.
2.  Run the script from your terminal:
    ```bash
    python generate_indicators.py
    ```
3.  The script will generate a new file named 'ohlcv_with_indicators.csv' containing the original data with all the new indicator columns.