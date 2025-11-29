# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:48:05 2025

@author: Enrico
"""

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader.data import DataReader


def build_market_info():
    raw = [
        ("BITCOIN", "Digital Currencies"),
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
        ("REIT_ASIA", "Real Estate Investment Trusts (REIT)")
    ]
    df = pd.DataFrame(raw, columns=["symbol (acronym)", "market_group"])
    df["Currency"] = "USD"
    df.loc[df["symbol (acronym)"] == "DAX", "Currency"] = "EUR"
    df.loc[df["symbol (acronym)"] == "CAC40", "Currency"] = "EUR"
    return df

market_info = build_market_info()

def normalize_columns(df):
    """Normalize column names: lowercase, strip spaces, replace non-alphanum with underscores."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r'\s+', '_', regex=True)
                  .str.replace(r'[^\w]', '', regex=True)
    )
    return df

def _dedupe_ohlcv(df):
    """
    Remove duplicate dates in OHLCV data, keeping the last occurrence.
    Assumes 'date' column exists.
    """
    if 'date' not in df.columns:
        raise ValueError("DataFrame must have a 'date' column")
    df = df.sort_values('date')  # ensure chronological order
    df = df.drop_duplicates(subset='date', keep='last')
    df = df.set_index('date')
    return df


def get_exchange_rate(currency, start_date, end_date):
    """Get daily exchange rate to USD."""
    if currency.upper() == 'USD':
        df = pd.DataFrame({
            'Date': pd.date_range(start=start_date, end=end_date),
            'Rate_to_USD': 1.0
        })
        return df
    fx_map = {
        'EUR': ('EURUSD=X', 1),
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
        'SGD': ('USDSGD=X', -1)
    }

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
    """
    Merge df with exchange rates and compute Value_USD.
    Ensures no MultiIndex issues.
    """
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
    """Fetch daily interpolated US CPI from FRED."""
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
    df = df.copy()
    df['Date'] = df.index
    inflation_index['Date'] = pd.to_datetime(inflation_index['Date']).dt.tz_localize(None)
    merged = df.merge(inflation_index, on='Date', how='left')
    latest_cpi = inflation_index['CPI_USD'].iloc[-1]
    merged['real_close'] = merged['Value_USD'] * (latest_cpi / merged['CPI_USD'])
    merged = merged.set_index('Date')
    return merged

def euclidean_vector(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Vectors must be same length")
    return float(np.sqrt(np.sum((a - b) ** 2)))

def chi_square_test(observed, expected):
    observed = np.array(observed, dtype=float)
    expected = np.array(expected, dtype=float)
    if observed.shape != expected.shape:
        raise ValueError("Length mismatch")
    if np.any(expected == 0):
        raise ValueError("Zero expected freq not allowed")
    chi_sq = np.sum((observed - expected) ** 2 / expected)
    return chi_sq

def process_asset_file(filepath, asset_name, market_info):
    """
    Process a single asset CSV file:
    - Clean columns and normalize names
    - Ensure date column exists
    - Deduplicate OHLCV data
    - Convert to USD if necessary
    - Adjust for inflation
    """
    # Load CSV
    df = pd.read_csv(filepath, on_bad_lines="skip", low_memory=False)

    # Normalize columns
    df = normalize_columns(df)

    # Find date-like column
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

    # Deduplicate OHLCV
    df = _dedupe_ohlcv(df)

    # Skip if empty
    if df.empty:
        print(f"Skipping {asset_name}: no valid dates after cleaning")
        return pd.DataFrame()

    # Get currency
    row = market_info.loc[market_info['symbol (acronym)'] == asset_name]
    currency = row['Currency'].iloc[0] if len(row) else 'USD'

    # Convert to USD if needed
    if currency != "USD":
        fx = get_exchange_rate(currency, df.index.min(), df.index.max())
        df = convert_to_usd(df, currency, fx)
    else:
        df['Value_USD'] = df.get('close', np.nan)

    # Inflation adjustment
    inflation_index = get_inflation_index(df.index.min(), df.index.max())
    df = adjust_for_inflation(df, inflation_index)

    return df



def looks_like_returns_matrix(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        try: df.index = pd.to_datetime(df.index)
        except: return False
    if not df.apply(lambda s: pd.api.types.is_numeric_dtype(s)).all():
        return False
    if not df.columns.str.contains(r"^\d").any():
        return False
    return True

def clean_edhec_returns(df, source):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def clean_names_and_dtypes(df, source, symbol_hint=None):
    df = normalize_columns(df)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    return df

def clean_any(df, source, symbol_hint=None):
    if looks_like_returns_matrix(df.rename(columns=str.lower)):
        return clean_edhec_returns(df, source)
    return clean_names_and_dtypes(df, source, symbol_hint)

def calendar_align(df):
    df = df.copy()

    # force datetime index
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    # localize/convert to UTC
    try:
        df.index = df.index.tz_localize("UTC")
    except TypeError:
        df.index = df.index.tz_convert("UTC")

    return df

def process_folder(path, market_info):
    results = {}
    for file in os.listdir(path):
        if not file.lower().endswith(".csv"): continue
        symbol = os.path.splitext(file)[0].upper()
        df = process_asset_file(os.path.join(path, file), symbol, market_info)
        df = calendar_align(df)
        results[symbol] = df
    return results

def summarize_data(df, symbol=None):
    summary = pd.DataFrame({
        "first_date": [df.index.min()],
        "last_date": [df.index.max()],
        "num_rows": [len(df)],
        "num_missing": [df.isna().sum().sum()],
        "columns": [", ".join(df.columns)]
    })
    if symbol:
        print(f"Summary for {symbol}:")
    print(summary)
    return summary

def plot_time_series(df, column="close", symbol=None):
    # Plot original vs real_close
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

    # Read dictionary
    dict_df = pd.read_csv(dictionary_csv, header=None)

    # Extract all tickers (columns 1..end)
    all_symbols = set()

    for _, row in dict_df.iterrows():
        # skip the first entry (category)
        symbols = row.dropna().tolist()[1:]
        symbols = [str(s).strip().upper() for s in symbols]
        all_symbols.update(symbols)

    # Convert symbols → filenames
    allowed_files = {f"{symbol}.csv" for symbol in all_symbols}

    print("Allowed files:", allowed_files)

    results = {}

    for file in os.listdir(path):
        if file not in allowed_files:
            continue  # skip files not in dictionary

        symbol = os.path.splitext(file)[0].upper()
        full_path = os.path.join(path, file)

        print(f"Processing {symbol} ...")

        df = process_asset_file(full_path, symbol, market_info)
        df = calendar_align(df)
        results[symbol] = df

    return results



import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
dictionary_path = os.path.join(script_dir, "company_dictionary.csv")
folder_path = os.path.join(script_dir, "by_the_stock")
dict_df = pd.read_csv(dictionary_path, header=None)




# folder_path = "full_history"

# dictionary_path = "company_dictionary.csv"



cleaned_data = process_folder_using_dictionary(
    folder_path,
    market_info,
    dictionary_path
)


for symbol, df in cleaned_data.items():
    print(symbol, df.head())

for symbol, df in cleaned_data.items():
    summarize_data(df, symbol)
    plot_time_series(df, "real_close", symbol)

output_folder = os.path.join(script_dir, "output_folder")
# output_folder = "output"
os.makedirs(output_folder, exist_ok=True)  # create folder if it doesn't exist

print("script_dir =", script_dir)
print("output_folder =", output_folder)
print("exists?", os.path.exists(output_folder))

for symbol, df in cleaned_data.items():

    full_path = os.path.join(output_folder, f"{symbol}_cleaned_full.csv")
    df.to_csv(full_path)
    
    df_2008_2020 = df.loc[(df.index >= "2008-01-01") & (df.index <= "2020-12-31")]
    path_2008_2020 = os.path.join(output_folder, f"{symbol}_cleaned_2008_2020.csv")
    df_2008_2020.to_csv(path_2008_2020)

    df_2020_onward = df.loc[df.index >= "2020-01-01"]
    path_2020_onward = os.path.join(output_folder, f"{symbol}_cleaned_2020_onward.csv")
    df_2020_onward.to_csv(path_2020_onward)

    print(f"Saved 3 versions for {symbol}")


for symbol, df in cleaned_data.items():
    # Save each cleaned DataFrame as CSV
    output_path = os.path.join(output_folder, f"{symbol}_cleaned.csv")
    df.to_csv(output_path)
    print(f"Saved cleaned data for {symbol} to {output_path}")
