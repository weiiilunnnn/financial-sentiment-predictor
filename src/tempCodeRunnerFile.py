from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

df = pd.read_csv("..\data\processed\MAYBANK_preprocessed.csv", parse_dates=['Date'])
df.set_index('Date', inplace=True)
# and you have features in X and target (e.g., 'Close' or returns) in y
X = df.drop(columns=['Close'])  # Example: dropping the target column
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.4,  # 40% test
    shuffle=False   # Keep time order (important for time series!)
)

rf_model = RandomForestRegressor(
    n_estimators=100,  # number of trees
    max_depth=None,    # allow full depth (or set based on tuning)
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")