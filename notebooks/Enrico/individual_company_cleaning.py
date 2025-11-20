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
    df['Date'] = df.index
    exchange_rates['Date'] = pd.to_datetime(exchange_rates['Date']).dt.tz_localize(None)
    merged = df.merge(exchange_rates, on='Date', how='left')
    merged['Value_USD'] = merged['close'] * merged['Rate_to_USD']
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
    df = pd.read_csv(filepath, on_bad_lines="skip", low_memory=False)
    date_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
    if date_col is None:
        raise ValueError(f"No date-like column found in {filepath}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    df = normalize_columns(df)
    if date_col != "date":
        df = df.rename(columns={date_col:"date"})
    df = df.set_index("date")
    df = _dedupe_ohlcv(df.reset_index()).set_index("date")

    # Currency to USD
    row = market_info.loc[market_info['symbol (acronym)']==asset_name]
    currency = row['Currency'].iloc[0] if len(row) else 'USD'

    if currency != "USD":
        fx = get_exchange_rate(currency, df.index.min(), df.index.max())
        df = convert_to_usd(df, currency, fx)
    else:
        df['Value_USD'] = df['close']

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


folder_path = "/content/unclean files"

cleaned_data = process_folder(folder_path, market_info)

for symbol, df in cleaned_data.items():
    print(symbol, df.head())

for symbol, df in cleaned_data.items():
    summarize_data(df, symbol)
    plot_time_series(df, "real_close", symbol)
