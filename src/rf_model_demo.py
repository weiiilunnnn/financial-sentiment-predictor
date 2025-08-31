import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
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

full_predictors = predictors + [
    "weekly_mean", "quarterly_mean", "annual_mean",
    "annual_weekly_mean", "annual_quarterly_mean",
    "open_close_ratio", "high_close_ratio", "low_close_ratio", "weekly_trend"
]

df = df.dropna()

# Split training for hyperparameter search
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
X_train = train_df[full_predictors]
y_train = train_df["Target"]

def run_random_search(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': [500, 700, 900, 1000],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 3, 5],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced']
    }
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='precision',
        cv=3,
        random_state=42,
        n_jobs=1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_params_

# Backtest with a given threshold
def backtest(data, model, predictors, threshold=0.48, start=200, step=250):
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

# Find the best threshold
def find_best_threshold(model, data, predictors, start=200, step=250):
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_threshold = 0.48
    best_score = 0
    for t in thresholds:
        preds = backtest(data, model, predictors, threshold=t, start=start, step=step)
        score = precision_score(preds["Target"], preds["Predictions"])
        if score > best_score:
            best_score = score
            best_threshold = t
    return best_threshold, best_score

if __name__ == "__main__":
    # Step 1: Hyperparameter tuning
    best_params = run_random_search(X_train, y_train)
    print("Best Hyperparameters:", best_params)

    # Step 2: Train Random Forest with best params
    best_rf_model = RandomForestClassifier(**best_params, random_state=42)

    # Step 3: Find the best threshold
    best_threshold, best_precision = find_best_threshold(best_rf_model, df, full_predictors)
    print(f"Best Threshold: {best_threshold:.2f} with Precision: {best_precision:.4f}")

    # Step 4: Backtest with optimized threshold
    predictions = backtest(df, best_rf_model, full_predictors, threshold=best_threshold)

    # Step 5: Evaluate
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    f1 = f1_score(predictions["Target"], predictions["Predictions"])
    print(f"Final Precision: {precision:.4f}")
    print(f"Final F1 Score: {f1:.4f}")
    print(f"Number of Buys: {predictions['Predictions'].sum()}")
