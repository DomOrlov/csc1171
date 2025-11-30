import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader.data import DataReader


def build_market_info():
    raw = [("BITCOIN", "Digital Currencies"),
        ("GOLD", "Commodities"),
        ("SILVER", "Commodities"),
        ("OIL", "Commodities"),
        ("VIX", "Volatility Index"),
        ("FTSE100", "Equities"),
        ("DAX", "Equities"),
        ("CAC40", "Equities"),
        ("NIKKEI", "Equities"),
        ("SP500", "Equities"),
        ("NASDAQ", "Equities"),
        ("REIT_USA", "Real Estate Investment Trusts (REIT)"),
        ("REIT_EUROPE", "Real Estate Investment Trusts (REIT)"),
        ("REIT_ASIA", "Real Estate Investment Trusts (REIT)")]
    
    df = pd.DataFrame(raw, columns=["symbol (acronym)", "market_group"])
    df["Currency"] = "USD"
    df.loc[df["symbol (acronym)"] == "DAX", "Currency"] = "EUR"
    df.loc[df["symbol (acronym)"] == "CAC40", "Currency"] = "EUR"
    return df

market_info = build_market_info()

def normalize_columns(df):
    # Normalize colums names
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r'\s+', '_', regex=True)
                  .str.replace(r'[^\w]', '', regex=True)
    )
    return df

def _dedupe_ohlcv(df):
    # Remove duplicate dates
    if 'date' not in df.columns:
        raise ValueError("DataFrame must have a 'date' column")
    df = df.sort_values('date')  # ensure chronological order
    df = df.drop_duplicates(subset='date', keep='last')
    df = df.set_index('date')
    return df


def get_exchange_rate(currency, start_date, end_date):
    # Get exchange rates for currencies
    if currency.upper() == 'USD':
        df = pd.DataFrame({
            'Date': pd.date_range(start=start_date, end=end_date),
            'Rate_to_USD': 1.0
        })
        return df
    fx_map = {'EUR': ('EURUSD=X', 1),
        'GBP': ('GBPUSD=X', 1),
        'AUD': ('AUDUSD=X', 1),
        'NZD': ('NZDUSD=X', 1),
        'CAD': ('USDCAD=X', -1),
        'JPY': ('USDJPY=X', -1),
        'CHF': ('USDCHF=X', -1),
        'CNY': ('USDCNY=X', -1),
        'HKD': ('USDHKD=X', -1),
        'SEK': ('USDSEK=X', -1),
        'NOK': ('USDNOK=X', -1),
        'SGD': ('USDSGD=X', -1)}

    cur = currency.upper()
    if cur in fx_map:
        symbol, invert = fx_map[cur]
    else:
        symbol, invert = (f"{cur}USD=X", 1)  # fallback

    data = yf.download(symbol, start=start_date, end=end_date)[['Close']].reset_index()
    data = data.rename(columns={'Close': 'Rate_to_USD'})
    if invert == -1:
        data['Rate_to_USD'] = 1 / data['Rate_to_USD']
    return data[['Date','Rate_to_USD']]

def convert_to_usd(df, currency, exchange_rates):
    # Convert to USD
    df = df.copy()

    # Reset index if it's a MultiIndex
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Ensure 'Date' column exists for merge
    if 'Date' not in df.columns:
        df['Date'] = df.index

    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    exchange_rates['Date'] = pd.to_datetime(exchange_rates['Date']).dt.tz_localize(None)

    merged = pd.merge(df, exchange_rates, on='Date', how='left')

    # Use 'close' if exists, otherwise fallback to first numeric column
    if 'close' in merged.columns:
        merged['Value_USD'] = merged['close'] * merged['Rate_to_USD']
    else:
        first_numeric = merged.select_dtypes(include='number').columns[0]
        merged['Value_USD'] = merged[first_numeric] * merged['Rate_to_USD']

    merged = merged.set_index('Date')
    return merged


def get_inflation_index(start_date, end_date):
    # Get inflation rate
    try:
        cpi = DataReader('CPIAUCNS', 'fred', start_date - pd.DateOffset(days=31), end_date)
    except Exception as e:
        print("FRED CPI fetch failed:", e)
        return pd.DataFrame({'Date': pd.date_range(start=start_date, end=end_date), 'CPI_USD': np.nan})

    cpi = cpi.reset_index().rename(columns={'DATE':'Date','CPIAUCNS':'CPI_USD'})
    cpi['Date'] = pd.to_datetime(cpi['Date'])

    daily_index = pd.DataFrame({'Date': pd.date_range(start=start_date, end=end_date)})
    daily_index = pd.merge_asof(daily_index.sort_values('Date'),
                                cpi.sort_values('Date'),
                                on='Date',
                                direction='backward')
    daily_index['CPI_USD'] = daily_index['CPI_USD'].ffill().bfill().interpolate(method='linear')
    return daily_index

def adjust_for_inflation(df, inflation_index):
    # Adjust the value for inflation
    df = df.copy()
    df['Date'] = df.index
    inflation_index['Date'] = pd.to_datetime(inflation_index['Date']).dt.tz_localize(None)
    merged = df.merge(inflation_index, on='Date', how='left')
    latest_cpi = inflation_index['CPI_USD'].iloc[-1]
    merged['real_close'] = merged['Value_USD'] * (latest_cpi / merged['CPI_USD'])
    merged = merged.set_index('Date')
    return merged

def process_asset_file(filepath, asset_name, market_info):
    # Processing pipeline

    # Load CSV
    df = pd.read_csv(filepath, on_bad_lines="skip", low_memory=False)

    # Normalize columns
    df = normalize_columns(df)

    # Find date column
    date_col = next((c for c in df.columns if "date" in c or "time" in c), None)
    if date_col is None:
        raise ValueError(f"No date-like column found in {filepath}")

    # Rename to 'date'
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')

    # Remove duplicates
    df = _dedupe_ohlcv(df)

    # Skip if empty file
    if df.empty:
        print(f"Skipping {asset_name}: no valid dates after cleaning")
        return pd.DataFrame()

    # Get currency for the asset
    row = market_info.loc[market_info['symbol (acronym)'] == asset_name]
    currency = row['Currency'].iloc[0] if len(row) else 'USD'

    # Convert to USD (if needed)
    if currency != "USD":
        fx = get_exchange_rate(currency, df.index.min(), df.index.max())
        df = convert_to_usd(df, currency, fx)
    else:
        df['Value_USD'] = df.get('close', np.nan)

    # Adjust for inflation
    inflation_index = get_inflation_index(df.index.min(), df.index.max())
    df = adjust_for_inflation(df, inflation_index)

    return df


def calendar_align(df):
    df = df.copy()

    # Force datetime index
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    # Convert to UTC
    try:
        df.index = df.index.tz_localize("UTC")
    except TypeError:
        df.index = df.index.tz_convert("UTC")

    return df

def process_folder(path, market_info):
    # Runs through files in a folder
    results = {}
    for file in os.listdir(path):
        if not file.lower().endswith(".csv"): continue
        symbol = os.path.splitext(file)[0].upper()
        df = process_asset_file(os.path.join(path, file), symbol, market_info)
        df = calendar_align(df)
        results[symbol] = df
    return results

def summarize_data(df, symbol=None):
    #Give a summary of processed data
    summary = pd.DataFrame({"first_date": [df.index.min()],
        "last_date": [df.index.max()],
        "num_rows": [len(df)],
        "num_missing": [df.isna().sum().sum()],
        "columns": [", ".join(df.columns)]})
    if symbol:
        print(f"Summary for {symbol}:")
    print(summary)
    return summary

def plot_time_series(df, column="close", symbol=None):
    # Plot original vs new close
    if "close" in df.columns and "real_close" in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df.index, df["close"], label="Original Close / AdjClose", alpha=0.8)
        plt.plot(df.index, df["real_close"], label="Inflation Adjusted Close", alpha=0.8)
        plt.title(f"{symbol} Close vs Inflation-Adjusted Close")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"{symbol}: missing 'close' or 'real_close', cannot plot comparison")

def process_folder_using_dictionary(path, market_info, dictionary_csv):
    # Run only for the top companies per market sector
    
    # Read dictionary
    dict_df = pd.read_csv(dictionary_csv, header=None)

    # Extract all tickers
    all_symbols = set()

    for _, row in dict_df.iterrows():
        
        # Skip the first entry (market sector)
        symbols = row.dropna().tolist()[1:]
        symbols = [str(s).strip().upper() for s in symbols]
        all_symbols.update(symbols)

    # Convert symbols to filenames
    allowed_files = {f"{symbol}.csv" for symbol in all_symbols}

    print("Allowed files:", allowed_files)

    results = {}

    for file in os.listdir(path):
        if file not in allowed_files:
            continue  # Skip files not in dictionary

        symbol = os.path.splitext(file)[0].upper()
        full_path = os.path.join(path, file)

        print(f"Processing {symbol}")

        df = process_asset_file(full_path, symbol, market_info)
        df = calendar_align(df)
        results[symbol] = df

    return results

folder_path = "archive (1)/full_history" # Change this to folder path containing data

dictionary_path = "archive (1)\company_dictionary.csv" # Change this to folder path containing company dictionary file

cleaned_data = process_folder_using_dictionary(folder_path,
    market_info, dictionary_path)

for symbol, df in cleaned_data.items():
    print(symbol, df.head())

for symbol, df in cleaned_data.items():
    summarize_data(df, symbol)
    plot_time_series(df, "real_close", symbol)


output_folder = "output"
os.makedirs(output_folder, exist_ok=True)  # create folder if it doesn't exist

for symbol, df in cleaned_data.items():
    
    full_path = os.path.join(output_folder, f"{symbol}_cleaned_full.csv")
    df.to_csv(full_path)

    print(f"Saved cleaned version for {symbol}")
