import os
import pandas as pd
import yfinance as yf

def fetch_stock_data(ticker, period="1y"):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period=period)

    if df.empty:
        print(f"[ERROR] No data found for {ticker}. Skipping.")
        return None
    return df

def save_stock_data(df, filepath):
    # Ensure correct column order: Open, High, Low, Close, Volume
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Save with Date as a column (not index)
    df.reset_index(inplace=True)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

def load_stock_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Check if file is empty
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"File is empty: {filepath}")

    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    return df

if __name__ == "__main__":
    ticker = "1155.KL"
    path = f"data/raw/{ticker}.csv"

    df = fetch_stock_data(ticker)
    if df is not None:
        save_stock_data(df, path)

        try:
            df = load_stock_data(path)
            print(df.head())
        except ValueError as e:
            print(f"[ERROR] {e}")