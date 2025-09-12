"""
Author: [M.fadakar]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Read and preprocess the data
# ------------------------------
df = pd.read_csv("pricebitcoin.csv")

# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# ------------------------------
# 2. Calculate 30-day logarithmic returns
# Formula: ln(P_t / P_{t-30})
# ------------------------------
df['Log_Return_30'] = np.nan
df.loc[30:, 'Log_Return_30'] = np.log(df['close'].iloc[30:].values / df['close'].iloc[:-30].values)

# Select required columns
result_df = df[['Date', 'Log_Return_30']].copy()

# Round to 4 decimal places and save
result_df['Log_Return_30'] = result_df['Log_Return_30'].round(4)
result_df.to_csv("lreturns30.csv", index=False)
print("✅ Calculations completed. Results saved in 'lreturns30.csv'.")

# ------------------------------
# 3. Descriptive statistics
# ------------------------------
log_returns = result_df['Log_Return_30'].dropna()

stats = log_returns.describe()
print("\n📊 Descriptive Statistics for 30-Day Log Returns:")
print(f"Count: {stats['count']:.0f}")
print(f"Mean: {stats['mean']:.4f}")
print(f"Std Dev: {stats['std']:.4f}")
print(f"Min: {stats['min']:.4f}")
print(f"25%: {stats['25%']:.4f}")
print(f"Median: {stats['50%']:.4f}")
print(f"75%: {stats['75%']:.4f}")
print(f"Max: {stats['max']:.4f}")

print("\n📈 Distribution Analysis:")
print(f"Skewness: {log_returns.skew():.4f}")
print(f"Kurtosis: {log_returns.kurtosis():.4f}")

# ------------------------------
# 4. Histogram of log returns
# ------------------------------
plt.figure(figsize=(10, 6), dpi=300)
log_returns.hist(bins=50, edgecolor='black', color='black')

plt.title("Histogram of 30-Day Log Returns", fontsize=14, color="black")
plt.xlabel("Log Return", fontsize=12, color="black")
plt.ylabel("Frequency", fontsize=12, color="black")

# Style adjustments
plt.gca().set_facecolor("white")
plt.gcf().patch.set_facecolor("white")
for spine in plt.gca().spines.values():
    spine.set_color("black")
plt.tick_params(colors="black")
plt.grid(False)

plt.tight_layout()
plt.show()

# ------------------------------
# 5. Time series plot (2012–2024)
# ------------------------------
filtered_df = result_df[(result_df['Date'].dt.year >= 2012) & (result_df['Date'].dt.year <= 2024)].copy()

plt.figure(figsize=(12, 8), dpi=300)
plt.plot(filtered_df['Date'], filtered_df['Log_Return_30'], color="black", linewidth=1)

plt.title("30-Day Log Returns of Bitcoin (2012–2024)", fontsize=16, color="black")
plt.xlabel("Year", fontsize=14, color="black")
plt.ylabel("Log Returns", fontsize=14, color="black")

# Style adjustments
plt.gca().set_facecolor("white")
plt.gcf().patch.set_facecolor("white")
for spine in plt.gca().spines.values():
    spine.set_color("black")
plt.tick_params(colors="black")

# X-axis ticks by year
years = range(2012, 2025)
plt.xticks(
    [pd.Timestamp(f"{year}-01-01") for year in years],
    [str(year) for year in years],
    rotation=45,
    color="black"
)

plt.grid(False)
plt.tight_layout()
plt.show()

# ------------------------------
# Notes
# ------------------------------
print("\n📌 Distribution Note: The histogram shows the shape of the log returns distribution.")
print("Skewness and kurtosis provide insights into symmetry and tail heaviness.")
print("📌 Time Series Note: The second chart shows the 30-day log returns over time from 2012 to 2024.")
