import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, f1_score
import numpy as np

# Load the data
df = pd.read_csv("data/processed/MAYBANK_preprocessed.csv", parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Base predictors
predictors = ["Close", "Volume", "Open", "High", "Low", "Return", "MA5", "MA10", "Volatility"]

# Feature engineering
df["weekly_mean"] = df["Close"].rolling(7).mean() / df["Close"]
df["quarterly_mean"] = df["Close"].rolling(90).mean() / df["Close"]
df["annual_mean"] = df["Close"].rolling(365).mean() / df["Close"]
df["annual_weekly_mean"] = df["annual_mean"] / df["weekly_mean"]
df["annual_quarterly_mean"] = df["annual_mean"] / df["quarterly_mean"]
df["weekly_trend"] = df["Target"].shift(1).rolling(7).sum()
df["open_close_ratio"] = df["Open"] / df["Close"]
df["high_close_ratio"] = df["High"] / df["Close"]
df["low_close_ratio"] = df["Low"] / df["Close"]

# Full predictor set
full_predictors = predictors + [
    "weekly_mean", "quarterly_mean", "annual_mean",
    "annual_weekly_mean", "annual_quarterly_mean",
    "open_close_ratio", "high_close_ratio", "low_close_ratio", "weekly_trend"
]

# Drop missing values caused by rolling calculations
df = df.dropna()

# Backtest with a given threshold
def backtest(data, model, predictors, threshold=0.46, start=200, step=250):
    predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i]
        test = data.iloc[i:(i+step)]
        model.fit(train[predictors], train["Target"])
        preds = model.predict_proba(test[predictors])[:, 1]
        preds_bin = (preds > threshold).astype(int)
        combined = pd.DataFrame({
            "Target": test["Target"],
            "Predictions": preds_bin
        }, index=test.index)
        predictions.append(combined)
    return pd.concat(predictions)

if __name__ == "__main__":
    # Fixed best hyperparameters (from your previous run)
    best_params = {
        'n_estimators': 900,
        'min_samples_split': 5,
        'min_samples_leaf': 3,
        'max_features': 'log2',
        'max_depth': None,
        'class_weight': None
    }

    # Initialize the Random Forest model with the best hyperparameters
    model = RandomForestClassifier(**best_params, random_state=42)

    # Backtest with the best threshold found previously (0.45)
    predictions = backtest(df, model, full_predictors, threshold=0.45)

    # Evaluate performance
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    f1 = f1_score(predictions["Target"], predictions["Predictions"])
    print(f"Final Precision: {precision:.4f}")
    print(f"Final F1 Score: {f1:.4f}")
    print(f"Number of Buys: {predictions['Predictions'].sum()}")

# Check how many trades would have been made
trade_counts = predictions["Predictions"].value_counts()
print(f"Trade Counts:\n{trade_counts}")