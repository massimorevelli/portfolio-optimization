##############################################################################
# Fetches daily S&P100 stock prices (Yahoo Finance) and Fama–French 3 factors
# (Kenneth French's data library) for the period 2013–2019.
#
# Outputs:
# - prices.csv: daily adjusted close prices
# - ff3.csv: daily Fama–French 3 factors
#

import yfinance as yf
import pandas as pd
from pandas_datareader import data as web


start_date = "2013-01-01"
end_date = "2019-12-31"


# =========================
# STOCKS
# =========================

tickers = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN",
    "AVGO","AXP","BA","BAC","BK","BKNG","BLK","BMY","BRK-B","C",
    "CAT","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVX","DHR",
    "DIS","DUK","EMR","EXC","F","FDX","GD","GE","GILD",
    "GOOG","GS","HON","IBM","INTC","JNJ","JPM","KO",
    "LIN","LLY","LMT","LOW","MA","MDLZ","MDT","MET","META","MKTX",
    "MMM","MO","MRK","MS","MSFT","NEE","NFLX","NKE","ORCL","OXY",
    "PEP","PFE","PG","PM","QCOM","RTX","SCHW","SO","SPGI",
    "T","TGT","TMO","TMUS","TXN","UNH","UPS","USB","V",
    "VZ","WBA","WFC","WMT","XOM"
    ]


print(f"Downloading prices for {len(tickers)} tickers from {start_date} to {end_date}")


data = yf.download(tickers,
                   start=start_date,
                   end=end_date,
                   interval="1d",
                   auto_adjust=True,threads=True
                   )

prices = data["Close"]

prices.to_csv('prices.csv')


# =========================
# FAMA–FRENCH 3 FACTORS
# =========================

ff = web.DataReader("F-F_Research_Data_Factors_Daily",
                    "famafrench",
                    start=start_date,
                    end=end_date
                   )

ff3 = ff[0].copy()
ff3.index = pd.to_datetime(ff3.index)
ff3 = ff3.loc[start_date:end_date]

ff3 = ff3 / 100

ff3.to_csv("ff3.csv")
