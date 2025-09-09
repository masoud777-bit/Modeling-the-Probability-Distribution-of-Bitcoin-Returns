"""
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

"""

import logging
from typing import List

import numpy as np
import pandas as pd
import talib

# Configuration
INPUT_FILE = "ohlcv.csv"
OUTPUT_FILE = "ohlcv_with_indicators.csv"

# Set up logging 
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def read_data(path: str) -> pd.DataFrame:
    """Read CSV and validate required columns."""
    df = pd.read_csv(path)

    # Accept either 'date' or no date column; if present convert to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    required = {'open', 'high', 'low', 'close', 'volume'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # Ensure numeric types
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the list of indicators requested and return a new DataFrame containing them.

    """
    out = df.copy()
    open_ = out['open']
    high = out['high']
    low = out['low']
    close = out['close']
    volume = out['volume']

    # --- ROC (percent) ---
    out['ROC3'] = close.pct_change(periods=3) * 100
    out['ROC7'] = close.pct_change(periods=7) * 100
    out['ROC14'] = close.pct_change(periods=14) * 100
    out['ROC30'] = close.pct_change(periods=30) * 100
    out['ROC90'] = close.pct_change(periods=90) * 100

    # --- rolling std of close ---
    out['std3'] = close.rolling(window=3, min_periods=1).std()
    out['std7'] = close.rolling(window=7, min_periods=1).std()
    out['std14'] = close.rolling(window=14, min_periods=1).std()
    out['std30'] = close.rolling(window=30, min_periods=1).std()
    out['std90'] = close.rolling(window=90, min_periods=1).std()

    # --- Simple Moving Averages (SMA) ---
    sma_periods = [8, 13, 21, 34, 55, 100, 200]
    for p in sma_periods:
        out[f'SMA_{p}'] = talib.SMA(close, timeperiod=p)

    # --- Exponential Moving Averages (EMA) ---
    for p in sma_periods:
        out[f'EMA_{p}'] = talib.EMA(close, timeperiod=p)

    # --- Weighted Moving Averages (WMA) ---
    for p in sma_periods:
        out[f'WMA_{p}'] = talib.WMA(close, timeperiod=p)

    # --- MACD sets ---
    # The user asked for several MACD variants: (fast, slow, signal) chosen below.
    # We rely on talib.MACD which returns (macd, macdsignal, macdhist).
    macd_configs = [
        (8, 13, 5, '8'),   # MACD_8: fast=8, slow=13, signal=5
        (13, 21, 8, '13'), # MACD_13
        (21, 34, 13, '21'),# MACD_21
        (34, 55, 21, '34'),# MACD_34
        (55, 89, 34, '55') # MACD_55
    ]
    for fast, slow, signal, label in macd_configs:
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        out[f'MACD_{label}'] = macd
        out[f'MACD_signal_{label}'] = macd_signal
        out[f'MACD_hist_{label}'] = macd_hist

    # --- RSI ---
    rsi_periods = [5, 8, 13, 21, 34, 55, 89]
    for p in rsi_periods:
        out[f'RSI_{p}'] = talib.RSI(close, timeperiod=p)

    # --- MOMENTUM (talib.MOM) ---
    mom_periods = [5, 8, 13, 21, 34, 55, 89]
    for p in mom_periods:
        out[f'MOMENTUM_{p}'] = talib.MOM(close, timeperiod=p)

    # --- OBV ---
    out['OBV'] = talib.OBV(close, volume)

    # --- ATR ---
    atr_periods = [8, 13, 21, 34, 55]
    for p in atr_periods:
        out[f'ATR_{p}'] = talib.ATR(high, low, close, timeperiod=p)

    # --- Hilbert Transform Trendline ---
    out['HT_TRENDLINE'] = talib.HT_TRENDLINE(close)

    # --- Average Price ---
    out['AVGPRICE'] = talib.AVGPRICE(open_, high, low, close)

    # --- BETA (volatility relationship) ---
    beta_periods = [3, 5, 8, 13, 21]
    for p in beta_periods:
        # talib.BETA expects (high, low, timeperiod)
        try:
            out[f'BETA_{p}'] = talib.BETA(high, low, timeperiod=p)
        except Exception:
            # If talib.BETA is not available on some systems, calculate a simple proxy: rolling covariance of close and high
            out[f'BETA_{p}'] = close.rolling(window=p).cov(high).div(close.rolling(window=p).var())

    # --- WMA on Volume ---
    vol_periods = [5, 8, 13, 21, 34, 55, 89]
    for p in vol_periods:
        out[f'WMA_volume_{p}'] = talib.WMA(volume, timeperiod=p)

    # --- Awesome Oscillator (AO) ---
    # AO = 5-period SMA(median_price) - 34-period SMA(median_price)
    median_price = (high + low) / 2.0
    out['AO'] = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()

    # --- Stochastic (K,D) for different K periods ---
    # We'll follow the slowk/slowd pattern used in the original code snippet.
    stoch_configs = [
        (8, 2, 2, '8'),
        (13, 3, 3, '13'),
        (21, 5, 5, '21'),
        (34, 8, 8, '34'),
        (55, 13, 13, '55')
    ]
    for k_period, slowk, slowd, label in stoch_configs:
        # talib.STOCH returns fastk, fastd (but with provided slow parameters it returns slowk, slowd)
        fastk, slowd_arr = talib.STOCH(high, low, close,
                                       fastk_period=k_period,
                                       slowk_period=slowk,
                                       slowk_matype=0,
                                       slowd_period=slowd,
                                       slowd_matype=0)
        # The talib return order may be (slowk, slowd) depending on parameters; we map them to K_x and D_x.
        out[f'K_{label}'] = fastk
        out[f'D_{label}'] = slowd_arr

    # --- DEMA ---
    dema_periods = [5, 8, 13, 21, 34, 55, 89]
    for p in dema_periods:
        out[f'DEMA_{p}'] = talib.DEMA(close, timeperiod=p)

    
    requested_cols: List[str] = [
        'open', 'high', 'low', 'close', 'volume',
        'ROC3', 'ROC7', 'ROC14', 'ROC30', 'ROC90',
        'std3', 'std7', 'std14', 'std30', 'std90',
        'SMA_8', 'SMA_13', 'SMA_21', 'SMA_34', 'SMA_55', 'SMA_100', 'SMA_200',
        'EMA_8', 'EMA_13', 'EMA_21', 'EMA_34', 'EMA_55', 'EMA_100', 'EMA_200',
        'WMA_8', 'WMA_13', 'WMA_21', 'WMA_34', 'WMA_55', 'WMA_100', 'WMA_200',
        'MACD_8', 'MACD_signal_8', 'MACD_hist_8',
        'MACD_13', 'MACD_signal_13', 'MACD_hist_13',
        'MACD_21', 'MACD_signal_21', 'MACD_hist_21',
        'MACD_34', 'MACD_signal_34', 'MACD_hist_34',
        'MACD_55', 'MACD_signal_55', 'MACD_hist_55',
        'RSI_5', 'RSI_8', 'RSI_13', 'RSI_21', 'RSI_34', 'RSI_55', 'RSI_89',
        'MOMENTUM_5', 'MOMENTUM_8', 'MOMENTUM_13', 'MOMENTUM_21', 'MOMENTUM_34', 'MOMENTUM_55', 'MOMENTUM_89',
        'OBV', 'ATR_8', 'ATR_13', 'ATR_21', 'ATR_34', 'ATR_55',
        'HT_TRENDLINE', 'AVGPRICE',
        'BETA_3', 'BETA_5', 'BETA_8', 'BETA_13', 'BETA_21',
        'WMA_volume_5', 'WMA_volume_8', 'WMA_volume_13', 'WMA_volume_21', 'WMA_volume_34', 'WMA_volume_55', 'WMA_volume_89',
        'AO', 'K_8', 'D_8', 'K_13', 'D_13', 'K_21', 'D_21', 'K_34', 'D_34', 'K_55', 'D_55',
        'DEMA_5', 'DEMA_8', 'DEMA_13', 'DEMA_21', 'DEMA_34', 'DEMA_55', 'DEMA_89'
    ]

    # Some columns (SMA_100/200 etc.) could be NaN if not enough data; they still must appear.
    # Filter to keep requested columns (if any are missing raise clear error).
    missing_requested = [c for c in requested_cols if c not in out.columns]
    if missing_requested:
        raise RuntimeError(f"Indicator computation failed: missing columns {missing_requested}")

    result = out[requested_cols]
    return result


def main():
    logger.info("Reading input file: %s", INPUT_FILE)
    df = read_data(INPUT_FILE)

    logger.info("Computing indicators...")
    df_ind = compute_indicators(df)

    logger.info("Saving output to %s", OUTPUT_FILE)
    df_ind.to_csv(OUTPUT_FILE, index=False)

    rows, cols = df_ind.shape
    logger.info("Finished. Output shape: %d rows x %d columns", rows, cols)


if __name__ == '__main__':
    main()
