"""
Options Entry Tracker — Streamlit App (Pro)

Adds to the starter:
- Delta-based option selection (BS greeks using chain IV)
- IV Rank & IV Percentile (proxy using 1Y realized volatility percentile + current chain IV context)
- Discord alerts when a signal triggers (use st.secrets or sidebar input)
- Richer backtest with ATR-based stops/targets and simple options P/L simulation (BS greeks path)
- One-click exports: CSV for trades/signals and a compact PDF report

How to run locally
1) Save as: options_entry_tracker.py
2) Create requirements.txt:
   streamlit\nyfinance\npandas\nnumpy\nmatplotlib\nrequests\nfpdf
3) Launch: streamlit run options_entry_tracker.py

Notes
- Educational tool. Not financial advice. Data from Yahoo via yfinance; accuracy not guaranteed.
- "IV Rank/Percentile" here use a *proxy* because historical option-IV isn’t reliably available via yfinance.
  We compute: (a) current chain IV stats (ATM IV, cross‑sectional percentile today), and (b) 1‑year *realized* volatility
  percentile vs its own 1‑year range. Treat as context, not a substitute for true IVR/IVP.
"""

from __future__ import annotations
import math
import datetime as dt
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import requests
from fpdf import FPDF

# ---------------------- Utility & Config
st.set_page_config(page_title="Options Entry Tracker (Pro)", layout="wide")

RISK_FREE = 0.045  # rough annualized risk-free for BS; change as needed
TODAY = dt.date.today()

# ---------------------- Data Fetch
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df.rename(columns={"Adj Close": "AdjClose"})
    df.dropna(inplace=True)
    return df

# ---------------------- Indicators

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean()
    out["EMA200"] = out["Close"].ewm(span=200, adjust=False).mean()
    # RSI(14)
    delta = out["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    out["RSI14"] = 100 - (100 / (1 + rs))
    # MACD (12,26,9)
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACDsig"] = out["MACD"].ewm(span=9, adjust=False).mean()
    # ATR(14)
    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - out["Close"].shift()).abs()
    low_close = (out["Low"] - out["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()
    # Donchian 20
    out["DonHigh20"] = out["High"].rolling(20).max()
    out["DonLow20"] = out["Low"].rolling(20).min()
    return out.dropna()

# ---------------------- Signals

def signal_trend(row) -> int:
    score = 0
    try:
        close = float(row["Close"])
        ema20 = float(row["EMA20"])
        ema50 = float(row["EMA50"])
        ema200 = float(row["EMA200"])
        rsi = float(row["RSI14"])
        macd = float(row["MACD"])
        macd_sig = float(row["MACDsig"])

        if close > ema20: score += 1
        if close > ema50: score += 1
        if ema50 > ema200: score += 1
        if rsi > 55: score += 1
        if macd > macd_sig: score += 1
    except Exception:
        score = 0
    return score

def signal_breakout(row) -> int:
    score = 0
    try:
        close = float(row["Close"])
        donhigh = float(row["DonHigh20"])
        rsi = float(row["RSI14"])
        macd = float(row["MACD"])
        macd_sig = float(row["MACDsig"])

        if close > donhigh: score += 3
        if rsi > 60: score += 1
        if macd > macd_sig: score += 1
    except Exception:
        score = 0
    return score

def signal_pullback(row) -> int:
    score = 0
    try:
        close = float(row["Close"])
        ema50 = float(row["EMA50"])
        rsi = float(row["RSI14"])
        macd = float(row["MACD"])
        macd_sig = float(row["MACDsig"])

        if close > ema50: score += 2
        if 40 <= rsi <= 55: score += 2
        if macd > macd_sig: score += 1
    except Exception:
        score = 0
    return score

def signal_meanrev(row) -> int:
    score = 0
    try:
        close = float(row["Close"])
        ema50 = float(row["EMA50"])
        rsi = float(row["RSI14"])
        macd = float(row["MACD"])
        macd_sig = float(row["MACDsig"])

        if close < ema50: score += 1
        if rsi < 30: score += 3
        if macd < macd_sig: score += 1
    except Exception:
        score = 0
    return score

# ---------------------- Volatility & IV Rank (Proxy)
@st.cache_data(show_spinner=False)
def realized_vol(df: pd.DataFrame, window: int = 21) -> pd.Series:
    # annualized realized vol from daily returns
    r = df["Close"].pct_change()
    rv = r.rolling(window).std() * np.sqrt(252)
    return rv

@st.cache_data(show_spinner=False)
def iv_context(ticker: str) -> dict:
    """Return today's chain IV stats (ATM IV, cross-sectional IV percentile) and
    realized vol percentile over 1y as a proxy for IVR/IVP.
    """
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    expirations = tk.options
    if not expirations:
        return {"spot": spot, "have_chain": False}
    # pick near 30D
    target = 30
    def pick_exp(exps):
        today = dt.date.today()
        best = None
        bestd = 10**9
        for e in exps:
            d = dt.datetime.strptime(e, "%Y-%m-%d").date()
            dte = (d - today).days
            if dte <= 0: continue
            if abs(dte - target) < bestd:
                bestd = abs(dte - target)
                best = e
        return best or exps[0]
    expiry = pick_exp(expirations)
    oc = tk.option_chain(expiry)
    calls, puts = oc.calls, oc.puts
    # ATM selection
    calls = calls.copy(); puts = puts.copy()
    for df_ in (calls, puts):
        df_["mid"] = (df_["bid"].fillna(0)+df_["ask"].fillna(0))/2
    allops = pd.concat([calls.assign(type="C"), puts.assign(type="P")])
    allops["dist"] = (allops["strike"] - spot).abs()
    atm = allops.sort_values("dist").head(10)
    atm_iv_mean = float(atm["impliedVolatility"].dropna().mean()) if not atm.empty else np.nan
    # cross-sectional IV percentile today
    cross_iv = allops["impliedVolatility"].dropna()
    if len(cross_iv) > 10 and not math.isnan(atm_iv_mean):
        pct_today = (cross_iv < atm_iv_mean).mean()
    else:
        pct_today = np.nan
    # realized vol percentile over 1y
    hist = tk.history(period="1y").dropna()
    rv = realized_vol(hist, window=21).dropna()
    if rv.empty:
        rv_pct = np.nan
    else:
        cur_rv = float(rv.iloc[-1])
        rv_min, rv_max = float(rv.min()), float(rv.max())
        rv_pct = (cur_rv - rv_min) / (rv_max - rv_min + 1e-12)
    return {
        "spot": float(spot),
        "have_chain": True,
        "expiry": expiry,
        "atm_iv_mean": atm_iv_mean,
        "iv_percentile_today": float(pct_today) if not math.isnan(pct_today) else np.nan,
        "rv_percentile_1y": float(rv_pct) if not math.isnan(rv_pct) else np.nan,
    }

# ---------------------- Black‑Scholes Greeks
from math import log, sqrt, exp
from scipy.stats import norm


def _bs_d1(S, K, r, sigma, T):
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T) + 1e-12)

def _bs_d2(d1, sigma, T):
    return d1 - sigma * math.sqrt(T)

def bs_delta(S, K, r, sigma, T, call=True):
    if T <= 0 or sigma <= 0:
        return 0.5 if abs(S-K) < 1e-6 else (1.0 if (call and S>K) else 0.0 if call else -1.0)
    d1 = _bs_d1(S, K, r, sigma, T)
    return norm.cdf(d1) if call else (norm.cdf(d1) - 1)

def bs_gamma(S, K, r, sigma, T):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = _bs_d1(S, K, r, sigma, T)
    return norm.pdf(d1) / (S * sigma * math.sqrt(T) + 1e-12)

def bs_theta(S, K, r, sigma, T, call=True):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = _bs_d1(S, K, r, sigma, T)
    d2 = _bs_d2(d1, sigma, T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    if call:
        return term1 - r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return term1 + r * K * math.exp(-r * T) * norm.cdf(-d2)

# ---------------------- Options Suggestion (Delta-based)
@dataclass
class OptionSuggestion:
    summary: str
    table: pd.DataFrame


def pick_expiration(expirations: List[str], target_dte: int) -> Optional[str]:
    today = dt.date.today()
    best = None
    bestd = 10**9
    for e in expirations:
        d = dt.datetime.strptime(e, "%Y-%m-%d").date()
        dte = (d - today).days
        if dte <= 0: continue
        if abs(dte - target_dte) < bestd:
            bestd = abs(dte - target_dte)
            best = e
    return best or (expirations[-1] if expirations else None)


def target_by_delta(df: pd.DataFrame, spot: float, r: float, T: float, call: bool, target_delta: float) -> pd.DataFrame:
    df = df.copy()
    iv = df["impliedVolatility"].fillna(df["impliedVolatility"].median()).replace([np.inf, -np.inf], np.nan).fillna(0.3)
    deltas = []
    for i, row in df.iterrows():
        K = float(row["strike"])
        s = float(iv.loc[i])
        d = bs_delta(spot, K, r, s, T, call=call)
        deltas.append(d)
    df["delta"] = deltas
    df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2
    df["spread"] = (df["ask"].fillna(0) - df["bid"].fillna(0))
    df["$OI"] = df["openInterest"].fillna(0)
    # choose closest to target delta, then prefer tighter spread & higher OI
    df["delta_diff"] = (df["delta"] - target_delta).abs()
    return df.sort_values(["delta_diff", "spread", "$OI"], ascending=[True, True, False])


def suggest_options(ticker: str, strategy: str, target_dte: int, delta_pick: float) -> OptionSuggestion:
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    exps = tk.options
    if not exps:
        return OptionSuggestion("No options available.", pd.DataFrame())
    expiry = pick_expiration(exps, target_dte)
    oc = tk.option_chain(expiry)
    calls, puts = oc.calls.copy(), oc.puts.copy()
    # time to expiry in years
    dte = max((dt.datetime.strptime(expiry, "%Y-%m-%d").date() - TODAY).days, 1)
    T = dte / 365.0

    rows = []
    if strategy in ("Long Call", "Call Credit Spread"):
        cand = target_by_delta(calls, spot, RISK_FREE, T, call=True, target_delta=delta_pick)
        if cand.empty:
            return OptionSuggestion("No matching calls.", pd.DataFrame())
        if strategy == "Long Call":
            rows.append(cand.head(3).assign(strategy="Long Call", expiry=expiry, spot=spot))
        else:
            short = cand.head(1)
            # long leg slightly higher strike (~ short strike + 1-3 steps)
            higher = calls[calls["strike"] > float(short.iloc[0]["strike"])].sort_values("strike").head(1)
            spread = pd.concat([short.assign(leg="short"), higher.assign(leg="long")])
            rows.append(spread.assign(strategy="Call Credit Spread", expiry=expiry, spot=spot))

    if strategy in ("Long Put", "Put Credit Spread", "Cash‑Secured Put"):
        tgt = -abs(delta_pick)
        candp = target_by_delta(puts, spot, RISK_FREE, T, call=False, target_delta=tgt)
        if candp.empty:
            return OptionSuggestion("No matching puts.", pd.DataFrame())
        if strategy == "Long Put":
            rows.append(candp.head(3).assign(strategy="Long Put", expiry=expiry, spot=spot))
        elif strategy == "Cash‑Secured Put":
            rows.append(candp.head(3).assign(strategy="Cash‑Secured Put", expiry=expiry, spot=spot))
        else:
            short = candp.head(1)
            lower = puts[puts["strike"] < float(short.iloc[0]["strike"])].sort_values("strike", ascending=False).head(1)
            spread = pd.concat([short.assign(leg="short"), lower.assign(leg="long")])
            rows.append(spread.assign(strategy="Put Credit Spread", expiry=expiry, spot=spot))

    table = pd.concat(rows).reset_index(drop=True) if rows else pd.DataFrame()
    return OptionSuggestion(f"Spot {ticker} ~ {spot:.2f} | Expiry {expiry} | Strategy {strategy}", table)

# ---------------------- Backtest (ATR stops & simple options P/L)
@dataclass
class BTConfig:
    signal_col: str
    trigger_at: int = 4
    hold_days: int = 5
    atr_mult_stop: float = 1.5
    atr_mult_target: float = 2.5
    direction: str = "long"  # long/short
    simulate_option: bool = True
    option_type: str = "call"  # call/put
    option_delta: float = 0.35

@dataclass
class BTResult:
    stats: dict
    trades: pd.DataFrame


def run_backtest(df: pd.DataFrame, cfg: BTConfig) -> BTResult:
    px = df["Close"].values
    atr = df["ATR14"].values
    sig = df[cfg.signal_col].values
    dates = df.index.to_list()

    # Find entries on threshold cross
    entries: List[int] = []
    for i in range(1, len(df)):
        if sig[i-1] < cfg.trigger_at and sig[i] >= cfg.trigger_at:
            entries.append(i)

    recs = []
    for i in entries:
        entry = px[i]
        entry_date = dates[i]
        stop = entry - cfg.atr_mult_stop * atr[i] if cfg.direction == "long" else entry + cfg.atr_mult_stop * atr[i]
        target = entry + cfg.atr_mult_target * atr[i] if cfg.direction == "long" else entry - cfg.atr_mult_target * atr[i]

        exit_price = None
        exit_date = None
        reason = "time"
        j_end = min(i + cfg.hold_days, len(px) - 1)
        for j in range(i + 1, j_end + 1):
            if cfg.direction == "long" and px[j] <= stop:
                exit_price, exit_date, reason = px[j], dates[j], "stop"
                break
            if cfg.direction == "long" and px[j] >= target:
                exit_price, exit_date, reason = px[j], dates[j], "target"
                break
            if cfg.direction == "short" and px[j] >= stop:
                exit_price, exit_date, reason = px[j], dates[j], "stop"
                break
            if cfg.direction == "short" and px[j] <= target:
                exit_price, exit_date, reason = px[j], dates[j], "target"
                break
        if exit_price is None:
            exit_price, exit_date = px[j_end], dates[j_end]

        ret_under = (exit_price - entry) / entry if cfg.direction == "long" else (entry - exit_price) / entry

        # Simple options P/L simulation: price path using BS greeks at entry (delta/gamma/theta constant approximation)
        opt_ret = np.nan
        if cfg.simulate_option:
            # Build an entry option (ATM-ish by delta) using entry info
            # For simplicity we use sigma from realized vol as proxy
            sigma = float((df["Close"].pct_change().rolling(21).std() * np.sqrt(252)).iloc[i])
            sigma = min(max(sigma, 0.05), 1.0)
            S = float(entry)
            call = cfg.option_type == "call"
            # choose strike from delta target
            def pick_strike_from_delta(S, call, target_delta):
                # naive search around S
                strikes = np.linspace(S*0.5, S*1.5, 101)
                T = cfg.hold_days/365
                best = strikes[0]
                bestd = 1e9
                for K in strikes:
                    d = bs_delta(S, K, RISK_FREE, sigma, T, call)
                    if abs(d - target_delta) < bestd:
                        bestd = abs(d - target_delta)
                        best = K
                return float(best)
            K = pick_strike_from_delta(S, call, cfg.option_delta if call else -abs(cfg.option_delta))
            T_total = cfg.hold_days/365
            delta0 = bs_delta(S, K, RISK_FREE, sigma, T_total, call)
            gamma0 = bs_gamma(S, K, RISK_FREE, sigma, T_total)
            theta0 = bs_theta(S, K, RISK_FREE, sigma, T_total, call)
            # approximate option value change over holding horizon using underlying move dS and average time decay
            dS = exit_price - entry
            dt_years = cfg.hold_days/365
            dOpt = delta0 * dS + 0.5 * gamma0 * (dS**2) - theta0 * dt_years
            # scale by an approximate starting premium (Black‑Scholes price proxy)
            # BS price proxy using put-call parity components
            d1 = _bs_d1(S, K, RISK_FREE, sigma, T_total)
            d2 = _bs_d2(d1, sigma, T_total)
            if call:
                prem = S*norm.cdf(d1) - K*math.exp(-RISK_FREE*T_total)*norm.cdf(d2)
            else:
                prem = K*math.exp(-RISK_FREE*T_total)*norm.cdf(-d2) - S*norm.cdf(-d1)
            prem = max(prem, 0.25)  # floor to avoid crazy ratios
            opt_ret = dOpt / prem

        recs.append({
            "EntryDate": entry_date,
            "EntryPrice": entry,
            "ExitDate": exit_date,
            "ExitPrice": exit_price,
            "Reason": reason,
            "RetUnderlying": ret_under,
            "RetOption": opt_ret,
        })

    trades = pd.DataFrame(recs)
    if trades.empty:
        stats = {"trades": 0, "win_rate": np.nan, "avg_ret": np.nan, "median_ret": np.nan, "avg_opt": np.nan}
    else:
        wins = (trades["RetUnderlying"] > 0).mean()
        stats = {
            "trades": len(trades),
            "win_rate": float(wins),
            "avg_ret": float(trades["RetUnderlying"].mean()),
            "median_ret": float(trades["RetUnderlying"].median()),
            "avg_opt": float(trades["RetOption"].dropna().mean()) if trades["RetOption"].notna().any() else np.nan,
        }
    return BTResult(stats=stats, trades=trades)

# ---------------------- Alerts (Discord)

def send_discord_alert(webhook_url: str, title: str, content: str) -> bool:
    try:
        payload = {"embeds": [{"title": title, "description": content}]}
        r = requests.post(webhook_url, json=payload, timeout=8)
        return r.status_code in (200, 204)
    except Exception:
        return False

# ---------------------- PDF Export

def export_pdf(summary: dict, trades: pd.DataFrame, path: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Options Entry Tracker — Report", ln=True, align='L')
    for k, v in summary.items():
        pdf.cell(200, 8, txt=f"{k}: {v}", ln=True, align='L')
    pdf.ln(4)
    # table (first 15 rows)
    cols = ["EntryDate","EntryPrice","ExitDate","ExitPrice","Reason","RetUnderlying","RetOption"]
    pdf.set_font("Arial", size=10)
    for _, row in trades.head(15)[cols].iterrows():
        pdf.cell(200, 6, txt=", ".join(str(row[c]) for c in cols), ln=True)
    pdf.output(path)
    return path

# ---------------------- UI
st.title("📈 Options Entry Tracker (Pro)")
st.caption("Entry timing research with delta-based picks, IV context, ATR backtests, alerts & exports.")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="QQQ").upper().strip()
    end = TODAY
    start = end - dt.timedelta(days=365 * 2)
    start = st.date_input("Start Date", value=start)
    end = st.date_input("End Date", value=end)

    st.subheader("Signals")
    model = st.selectbox("Model", ["SigTrend", "SigBreakout", "SigPullback", "SigMeanRev"], index=0)
    trigger_at = st.slider("Trigger threshold", 1, 5, 4)
    hold_days = st.slider("Hold days", 2, 30, 7)
    atr_stop = st.slider("ATR Stop (x)", 0.5, 5.0, 1.5, 0.1)
    atr_target = st.slider("ATR Target (x)", 0.5, 8.0, 2.5, 0.1)

    st.subheader("Options Picks")
    strategy = st.selectbox(
        "Strategy",
        ["Long Call", "Long Put", "Cash‑Secured Put", "Put Credit Spread", "Call Credit Spread"],
        index=0,
    )
    target_dte = st.slider("Target DTE (days)", 7, 90, 30)
    target_delta = st.slider("Target |delta|", 0.10, 0.60, 0.35, 0.05)

    st.subheader("Alerts")
    webhook_default = st.secrets.get("DISCORD_WEBHOOK", "") if hasattr(st, "secrets") else ""
    discord_webhook = st.text_input("Discord Webhook (optional)", value=webhook_default)
    enable_alerts = st.checkbox("Enable alert on today's trigger", value=False)

# Fetch & compute
if ticker:
    df = fetch_history(ticker, start, end)
    if df.empty:
        st.error("No data returned. Check ticker or dates.")
        st.stop()

    df = compute_indicators(df)
    df = add_signals(df)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"{ticker} Price & EMAs")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df["Close"], label="Close")
        ax.plot(df.index, df["EMA20"], label="EMA20")
        ax.plot(df.index, df["EMA50"], label="EMA50")
        ax.plot(df.index, df["EMA200"], label="EMA200")
        ax.set_xlabel("Date"); ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        st.pyplot(fig)

        st.subheader("RSI14 & MACD")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(df.index, df["RSI14"], label="RSI14")
        ax2.axhline(30, linestyle="--"); ax2.axhline(70, linestyle="--")
        ax2.set_ylabel("RSI")
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.plot(df.index, df["MACD"], label="MACD")
        ax3.plot(df.index, df["MACDsig"], label="Signal")
        ax3.set_ylabel("MACD")
        ax3.legend(loc="upper left")
        st.pyplot(fig3)

    with col2:
        st.subheader("Latest Readings (today)")
        latest_cols = ["Close","EMA20","EMA50","EMA200","RSI14","MACD","MACDsig","ATR14","DonHigh20","DonLow20","SigTrend","SigBreakout","SigPullback","SigMeanRev"]
        st.dataframe(df.iloc[-1][latest_cols].to_frame("value"))

    st.markdown("---")
    st.subheader("Volatility — IV Context (proxy)")
    ctx = iv_context(ticker)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot", f"{ctx.get('spot', float('nan')):.2f}")
    c2.metric("ATM IV (today)", f"{(ctx.get('atm_iv_mean', float('nan')) or float('nan'))*100:.1f}%")
    c3.metric("IV Cross‑Sec Percentile (today)", f"{(ctx.get('iv_percentile_today', float('nan')) or float('nan'))*100:.1f}%")
    c4.metric("Realized Vol Percentile (1y)", f"{(ctx.get('rv_percentile_1y', float('nan')) or float('nan'))*100:.1f}%")

    st.caption("IV rank/percentile shown with proxies; for true IVR/IVP use a historical options IV data provider.")

    st.markdown("---")
    st.subheader("Backtest — ATR Stops & Options P/L (approx)")
    direction = "long" if model in ("SigTrend","SigBreakout","SigPullback") else "short"
    opt_type = "call" if direction == "long" else "put"
    bt_cfg = BTConfig(signal_col=model, trigger_at=trigger_at, hold_days=hold_days, atr_mult_stop=atr_stop, atr_mult_target=atr_target, direction=direction, simulate_option=True, option_type=opt_type, option_delta=target_delta)
    bt_res = run_backtest(df, bt_cfg)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trades", bt_res.stats.get("trades", 0))
    m2.metric("Win Rate", f"{(bt_res.stats.get('win_rate', float('nan')) or 0)*100:.1f}%")
    m3.metric("Avg Underlying Ret", f"{(bt_res.stats.get('avg_ret', 0))*100:.2f}%")
    m4.metric("Median Underlying Ret", f"{(bt_res.stats.get('median_ret', 0))*100:.2f}%")
    m5.metric("Avg Option Ret (approx)", f"{(bt_res.stats.get('avg_opt', 0))*100:.2f}%")

    st.dataframe(bt_res.trades.tail(30))

    # Export buttons
    st.download_button("Download trades CSV", bt_res.trades.to_csv(index=False).encode(), file_name=f"{ticker}_trades.csv", mime="text/csv")

    # PDF export
    if st.button("Export PDF Report"):
        summary = {
            "Ticker": ticker,
            "Signal": model,
            "Threshold": trigger_at,
            "Hold days": hold_days,
            "ATR stop x": atr_stop,
            "ATR target x": atr_target,
            "Avg U Ret": f"{(bt_res.stats.get('avg_ret', 0))*100:.2f}%",
            "Avg Opt Ret": f"{(bt_res.stats.get('avg_opt', 0))*100:.2f}%",
        }
        path = export_pdf(summary, bt_res.trades, path="report.pdf")
        with open(path, "rb") as f:
            st.download_button("Download PDF report", f, file_name=f"{ticker}_report.pdf", mime="application/pdf")

    st.markdown("---")
    st.subheader("Options Chain Suggestions (delta‑based)")
    try:
        suggestion = suggest_options(ticker, strategy, target_dte, target_delta)
        st.caption(suggestion.summary)
        if suggestion.table.empty:
            st.warning("No suggestions available.")
        else:
            show_cols = [c for c in ["strategy","expiry","spot","contractSymbol","type","leg","strike","bid","ask","mid","spread","impliedVolatility","delta","$OI"] if c in suggestion.table.columns]
            st.dataframe(suggestion.table[show_cols].reset_index(drop=True))
    except Exception as e:
        st.error(f"Options lookup failed: {e}")

    # Alerts — if today crossed the threshold
    if enable_alerts and discord_webhook:
        last_sig = df.iloc[-1][model]
        prev_sig = df.iloc[-2][model]
        triggered = prev_sig < trigger_at <= last_sig
        if triggered:
            msg = f"{ticker} {model} crossed >= {trigger_at} on {df.index[-1].date()}\nPrice: {df.iloc[-1]['Close']:.2f} | RSI: {df.iloc[-1]['RSI14']:.1f} | MACD: {df.iloc[-1]['MACD']:.2f}"
            ok = send_discord_alert(discord_webhook, title=f"Signal Trigger: {ticker}", content=msg)
            st.success("Discord alert sent!" if ok else "Failed to send Discord alert.")
        else:
            st.info("No new trigger today.")

st.markdown(
    """
---
**Disclaimers**
- Educational use only. Trading options involves substantial risk and is not suitable for all investors.
- Backtests/greeks are simplified approximations (constant greeks across hold). Always verify with your broker tools.
- For true IV Rank/Percentile, use a data provider that offers historical option implied volatility per underlying.
    """
)
