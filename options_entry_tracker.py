"""
Options Entry Tracker — Streamlit App (Extended Version)

Includes: plotting, option evaluation, PDF export, Discord alert features.

Updated requirements:
- numpy
- pandas
- yfinance
- streamlit
- matplotlib (for plotting)
- fpdf (for PDF export)
- requests (for Discord webhooks)
- scipy (for statistical functions)

Note: The app should run in a standard Python environment (CPython 3.10+) and not in Pyodide/browser-based environments to avoid errors such as 'micropip not found'.
"""

from __future__ import annotations
import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import requests
from fpdf import FPDF
from scipy.stats import norm

st.set_page_config(page_title="Options Entry Tracker (Pro)", layout="wide")
RISK_FREE = 0.045  # annualized risk-free rate
TODAY = dt.date.today()

# ---------------------- Utility Functions

def safe_float(val, default=np.nan):
    try:
        return float(val)
    except Exception:
        return default

def ensure_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna().reset_index(drop=True)

# ---------------------- Technical Indicator Computations

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    delta = df['Close'].diff()
    up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df['RSI14'] = 100 - (100 / (1 + rs))
    df['DonHigh20'] = df['Close'].rolling(20).max()
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACDsig'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# ---------------------- Signal Functions

def signal_trend(row) -> int:
    score = 0
    close = safe_float(row.get("Close"))
    ema20 = safe_float(row.get("EMA20"))
    ema50 = safe_float(row.get("EMA50"))
    ema200 = safe_float(row.get("EMA200"))
    rsi = safe_float(row.get("RSI14"))
    macd = safe_float(row.get("MACD"))
    macd_sig = safe_float(row.get("MACDsig"))
    if np.isnan([close, ema20, ema50, ema200, rsi, macd, macd_sig]).any(): return 0
    if close > ema20: score +=1
    if close > ema50: score +=1
    if ema50 > ema200: score +=1
    if rsi > 55: score +=1
    if macd > macd_sig: score +=1
    return score

def signal_breakout(row) -> int:
    score = 0
    close = safe_float(row.get("Close"))
    donhigh = safe_float(row.get("DonHigh20"))
    rsi = safe_float(row.get("RSI14"))
    macd = safe_float(row.get("MACD"))
    macd_sig = safe_float(row.get("MACDsig"))
    if np.isnan([close, donhigh, rsi, macd, macd_sig]).any(): return 0
    if close > donhigh: score +=3
    if rsi>60: score +=1
    if macd>macd_sig: score +=1
    return score

def signal_pullback(row) -> int:
    score=0
    close=safe_float(row.get("Close"))
    ema50=safe_float(row.get("EMA50"))
    rsi=safe_float(row.get("RSI14"))
    macd=safe_float(row.get("MACD"))
    macd_sig=safe_float(row.get("MACDsig"))
    if np.isnan([close, ema50, rsi, macd, macd_sig]).any(): return 0
    if close>ema50: score+=2
    if 40<=rsi<=55: score+=2
    if macd>macd_sig: score+=1
    return score

def signal_meanrev(row) -> int:
    score=0
    close=safe_float(row.get("Close"))
    ema50=safe_float(row.get("EMA50"))
    rsi=safe_float(row.get("RSI14"))
    macd=safe_float(row.get("MACD"))
    macd_sig=safe_float(row.get("MACDsig"))
    if np.isnan([close, ema50, rsi, macd, macd_sig]).any(): return 0
    if close<ema50: score+=1
    if rsi<30: score+=3
    if macd<macd_sig: score+=1
    return score

# ---------------------- Add Signals Wrapper

def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    df=ensure_numeric(df, ["Close","EMA20","EMA50","EMA200","RSI14","MACD","MACDsig","DonHigh20"])
    df["SigTrend"] = df.apply(signal_trend, axis=1)
    df["SigBreakout"] = df.apply(signal_breakout, axis=1)
    df["SigPullback"] = df.apply(signal_pullback, axis=1)
    df["SigMeanRev"] = df.apply(signal_meanrev, axis=1)
    return df

# ---------------------- Plotting, Option Evaluation, PDF Export, Discord Alerts

def plot_stock(df, ticker):
    plt.figure(figsize=(10,4))
    plt.plot(df['Date'], df['Close'], label='Close')
    plt.plot(df['Date'], df['EMA20'], label='EMA20')
    plt.plot(df['Date'], df['EMA50'], label='EMA50')
    plt.legend()
    plt.title(f'{ticker} Price & EMAs')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(plt)

def evaluate_options(df):
    df['OptionScore'] = df[['SigTrend','SigBreakout','SigPullback','SigMeanRev']].sum(axis=1)
    return df

def export_pdf(df, filename='report.pdf'):
    pdf=FPDF()
    pdf.add_page()
    pdf.set_font('Arial','B',12)
    pdf.cell(0,10,'Options Tracker Report',0,1,'C')
    pdf.ln(5)
    for i,row in df.tail(10).iterrows():
        pdf.cell(0,8,f"{row['Date'].date()} | Close: {row['Close']:.2f} | Score: {row['OptionScore']}",0,1)
    pdf.output(filename)

def send_discord_alert(message, webhook_url):
    payload={'content':message}
    requests.post(webhook_url,json=payload)

# ---------------------- Streamlit App

def main():
    st.title('Options Entry Tracker (Pro)')
    ticker_input=st.text_input('Enter Stock Ticker','AAPL')
    if ticker_input:
        df=yf.download(ticker_input, period='6mo', interval='1d')
        df.reset_index(inplace=True)
        df=compute_indicators(df)
        df=add_signals(df)
        df=evaluate_options(df)
        st.dataframe(df.tail(20))
        plot_stock(df,ticker_input)

        if st.button('Export PDF'):
            export_pdf(df)
            st.success('PDF exported successfully')

        webhook_url=st.text_input('Discord Webhook URL')
        if st.button('Send Discord Alert') and webhook_url:
            latest_score=df['OptionScore'].iloc[-1]
            send_discord_alert(f'{ticker_input} latest option score: {latest_score}',webhook_url)
            st.success('Discord alert sent')

if __name__=='__main__':
    main()
