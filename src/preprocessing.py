# src/preprocessing.py

import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw stock price data.
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and ensure correct data types.
    """
    df = df.dropna()  # Remove missing rows if any

    # Dynamically select numeric columns that exist
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    numeric_cols = [col for col in expected_cols if col in df.columns]

    # Convert them to numeric
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return df.dropna()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features for model training.
    """
    # Daily returns
    df['Return'] = df['Close'].pct_change()

    # Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()

    # Volatility (rolling standard deviation)
    df['Volatility'] = df['Return'].rolling(window=5).std()

    # Shift target for next-day prediction (1 if price goes up, 0 if down)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df.dropna()

def preprocess_pipeline(filepath: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    """
    df = load_data(filepath)
    df = clean_data(df)
    df = add_features(df)
    return df

if __name__ == "__main__":
    raw_data_path = "data/raw/1155.KL.csv"  # Change to your actual file
    processed_df = preprocess_pipeline(raw_data_path)
    processed_df.to_csv("data/processed/MAYBANK_preprocessed.csv", index=False)
    print("Preprocessing completed. Saved to data/processed/")