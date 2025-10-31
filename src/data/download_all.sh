#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

mkdir -p data/raw/Global_Stock_Market_2008-2023
kaggle datasets download -d pavankrishnanarne/global-stock-market-2008-present \
  -p data/raw/Global_Stock_Market_2008-2023 --force
unzip -o data/raw/Global_Stock_Market_2008-2023/*.zip -d data/raw/Global_Stock_Market_2008-2023
rm -f data/raw/Global_Stock_Market_2008-2023/*.zip

mkdir -p data/raw/EDHEC_Hedge_Fund_Returns
kaggle datasets download -d petrirautiainen/edhec-hedge-fund-historical-return-index-series \
  -p data/raw/EDHEC_Hedge_Fund_Returns --force
unzip -o data/raw/EDHEC_Hedge_Fund_Returns/*.zip -d data/raw/EDHEC_Hedge_Fund_Returns
rm -f data/raw/EDHEC_Hedge_Fund_Returns/*.zip

mkdir -p data/raw/AMEX_NYSE_NASDAQ_stock_histories
kaggle datasets download -d qks1lver/amex-nyse-nasdaq-stock-histories \
  -p data/raw/AMEX_NYSE_NASDAQ_stock_histories --force
unzip -o data/raw/AMEX_NYSE_NASDAQ_stock_histories/*.zip -d data/raw/AMEX_NYSE_NASDAQ_stock_histories
rm -f data/raw/AMEX_NYSE_NASDAQ_stock_histories/*.zip

mkdir -p data/raw/SP500_ETF_FX_Crypto_Daily
kaggle datasets download -d benjaminpo/s-and-p-500-with-dividends-and-splits-daily-updated \
  -p data/raw/SP500_ETF_FX_Crypto_Daily --force
unzip -o data/raw/SP500_ETF_FX_Crypto_Daily/*.zip -d data/raw/SP500_ETF_FX_Crypto_Daily
rm -f data/raw/SP500_ETF_FX_Crypto_Daily/*.zip
