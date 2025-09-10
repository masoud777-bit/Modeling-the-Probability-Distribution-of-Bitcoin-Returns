# Feature engineering on technical data.py

generate_indicators.py

Purpose
-------
This script reads an OHLCV file and computes a set of technical indicators.

Required packages
-----------------
- pandas
- numpy
- TA-Lib (Python wrapper, e.g. `pip install TA-Lib`). On many systems TA-Lib C library must be installed first.

Usage
-----
1. Place your input CSV named `ohlcv.csv` in the same folder or change `INPUT_FILE` value.
2. Run the script: `python generate_indicators.py`
3. Output file will be saved as `ohlcv_with_indicators.csv`.

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
