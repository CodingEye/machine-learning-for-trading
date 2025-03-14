import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the CSV file
# Replace 'nasdaq_data.csv' with the actual file path if different
symbol = "USTEC"
df = pd.read_csv('ustec_historical.csv')

# Convert 'time' to datetime
df['time'] = pd.to_datetime(df['time'])

# Sort by time (just in case)
df = df.sort_values('time')

# Create target variable: 1 if next close > current close, 0 otherwise
df['next_close'] = df['close'].shift(-1)
df['price_up'] = (df['next_close'] > df['close']).astype(int)

# Drop rows where 'price_up' is NaN (last row)
df = df.dropna(subset=['price_up'])

# Feature engineering
# Close position within high-low range
df['close_position'] = np.where(df['high'] > df['low'],
                                (df['close'] - df['low']) / (df['high'] - df['low']),
                                0.5)

# Volume ratio relative to 20-period moving average
df['MA20_tick_volume'] = df['tick_volume'].rolling(window=20).mean()
df['volume_ratio'] = df['tick_volume'] / df['MA20_tick_volume']

# Time features
df['hour'] = df['time'].dt.hour
df['minute'] = df['time'].dt.minute

# Lagged features (1 to 5 periods)
for i in range(1, 6):
    df[f'close_position_lag{i}'] = df['close_position'].shift(i)
    df[f'volume_ratio_lag{i}'] = df['volume_ratio'].shift(i)

# Drop rows with NaN values (due to moving averages and lags)
df = df.dropna().reset_index(drop=True)

# Define feature list
features = ['close_position', 'volume_ratio', 'spread', 'hour', 'minute'] + \
           [f'close_position_lag{i}' for i in range(1, 6)] + \
           [f'volume_ratio_lag{i}' for i in range(1, 6)]

# Split data into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

X_train = train_df[features]
y_train = train_df['price_up']
X_test = test_df[features]
y_test = test_df['price_up']

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Prediction Accuracy: {accuracy:.4f}")

# Simulate trading strategy
test_df = test_df.copy()
test_df['strategy_return'] = np.where(y_pred == 1,
                                      (test_df['next_close'] - test_df['close']) / test_df['close'],
                                      0)
strategy_cumulative_return = (1 + test_df['strategy_return']).prod() - 1
print(f"Strategy Cumulative Return: {strategy_cumulative_return:.4%}")

# Calculate buy-and-hold return
buy_and_hold_return = (test_df['next_close'].iloc[-1] / test_df['close'].iloc[0]) - 1
print(f"Buy-and-Hold Cumulative Return: {buy_and_hold_return:.4%}")