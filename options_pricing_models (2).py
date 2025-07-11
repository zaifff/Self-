# Real-Life Tesla Option Pricing Project using Black-Scholes, MLP, and LSTM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# === Step 1: Download Tesla Historical Data ===
tesla = yf.Ticker("TSLA")
hist = tesla.history(period="1y")

# Calculate volatility (20-day rolling std dev of returns, annualized)
hist['Return'] = hist['Close'].pct_change()
hist['Volatility'] = hist['Return'].rolling(window=20).std() * np.sqrt(252)
hist = hist.dropna()

# === Step 2: Get Valid Option Expiry and Chain ===
options_dates = tesla.options
now = pd.Timestamp.now()

valid_expiry = None
for exp_str in options_dates:
    exp_date = pd.to_datetime(exp_str)
    T = (exp_date - now).total_seconds() / (365 * 24 * 60 * 60)
    if T > 0:
        valid_expiry = exp_str
        break

if valid_expiry is None:
    raise ValueError("No valid future expiry date found. Check option chain data.")

print(f"Using expiry date: {valid_expiry}")

opt_chain = tesla.option_chain(valid_expiry)
calls = opt_chain.calls.copy()

# Spot price and time to expiry
spot_price = hist['Close'].iloc[-1]
expiry_dt = pd.to_datetime(valid_expiry)
time_delta = expiry_dt - now
T = time_delta.total_seconds() / (365 * 24 * 60 * 60)

# Add model features
calls['S'] = spot_price
calls['T'] = T

historic_vol = hist['Volatility'].iloc[-1]
if historic_vol <= 0 or np.isnan(historic_vol):
    implied_vols = calls['impliedVolatility']
    sigma = implied_vols[implied_vols > 0].mean()
    print(f"Using fallback implied volatility: {sigma}")
else:
    sigma = historic_vol

calls['sigma'] = sigma
calls['r'] = 0.045  # risk-free rate

# Clean and filter data
calls = calls.dropna(subset=['lastPrice', 'strike', 'sigma'])
calls = calls[(calls['strike'] >= spot_price * 0.85) & (calls['strike'] <= spot_price * 1.15)]

if len(calls) == 0:
    raise ValueError("No calls left after filtering. Adjust strike range.")

# === Step 3: Black-Scholes Model ===
def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

calls['bs_price'] = calls.apply(lambda row: black_scholes(
    row['S'], row['strike'], row['T'], row['r'], row['sigma']), axis=1)

calls = calls.dropna(subset=['bs_price'])

# === Step 4: Prepare Data for ML Models ===
X = calls[['S', 'strike', 'T', 'sigma']].values
y = calls['lastPrice'].values
indices = np.arange(len(calls))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === MLP Model ===
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
mlp_preds = mlp.predict(X_test_scaled)

# === LSTM Model ===
X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_lstm, y, test_size=0.2, random_state=42
)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 4)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_lstm, y_train_lstm, epochs=50, verbose=0)
lstm_preds = model.predict(X_test_lstm).flatten()

# === Evaluation ===
bs_test = calls.iloc[idx_test]['bs_price'].values
print("MSE (Black-Scholes):", mean_squared_error(y_test, bs_test))
print("MSE (MLP):", mean_squared_error(y_test, mlp_preds))
print("MSE (LSTM):", mean_squared_error(y_test, lstm_preds))

# === Plot ===

# Plot Black-Scholes prediction vs true
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True Market Price', alpha=0.6)
plt.plot(bs_test, label='Black-Scholes Prediction', alpha=0.6)
plt.title('Black-Scholes vs Market Price')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True Market Price', alpha=0.6)
plt.plot(mlp_preds, label='MLP Prediction', alpha=0.6)
plt.plot(lstm_preds, label='LSTM Prediction', alpha=0.6)
plt.title('Tesla Call Option Price Predictions')
plt.legend()
plt.show()

# === Conclusion ===
print("""
Conclusion:
This project compares three option pricing models: Black-Scholes (analytical), MLP (feedforward neural network), and LSTM (sequence-based neural network).
- Black-Scholes serves as a theoretical baseline but assumes constant volatility.
- MLP and LSTM learn from market data patterns and may capture nonlinear relationships.
- By comparing Mean Squared Error (MSE), we identify which model most closely matches real market prices.
This demonstrates how traditional finance models and modern machine learning can complement each other in real-world financial predictions.
""")
