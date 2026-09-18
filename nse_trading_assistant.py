"""
NSE Daily Trading Research Assistant
=====================================

WHAT THIS IS
------------
A research tool that, every trading day before NSE market open, scans a
watchlist of stocks, runs technical analysis + news sentiment analysis,
and sends you ONE notification with its best-ranked idea for the day:
  - Suggested BUY price zone and BUY time window
  - Suggested TARGET price (projected, not guaranteed)
  - Suggested STOP-LOSS price
  - Suggested SELL-BY time
  - A short plain-English "why"

WHAT THIS IS NOT
----------------
- This does NOT guarantee any return. "Target %" is a projection based on
  volatility and technical structure, not a promise. Some days the honest
  output is "no clear setup today" -- the script WILL say that instead of
  forcing a pick, because forcing a low-quality pick every day is how
  accounts blow up.
- This does NOT place trades automatically. It only generates a signal.
  You review it and execute manually via your own broker if you choose to.
- This is not financial advice. You are responsible for your own trades.

HOW IT WORKS
------------
1. Pull ~3 months of daily OHLCV data per watchlist stock (yfinance).
2. Compute technical signals: RSI, MACD, volume spike ratio, ATR
   (average true range, used to size realistic targets/stop-loss),
   proximity to recent support/resistance.
3. Pull recent news per stock and score sentiment using the Claude API
   with web search.
4. Combine into a composite score per stock, rank them.
5. If the top score clears a quality bar -> build a signal with
   ATR-based target/stop-loss. If nothing clears the bar -> "no trade
   today" notification instead.
6. Push the result via ntfy.sh (free push notification app).
7. Log every pick to a local CSV so you can track real accuracy over
   time instead of trusting vibes.

COST: $0
--------
Every piece of this pipeline is free:
  - Price/volume data: yfinance (free, no key)
  - News sentiment: Google News RSS + keyword scoring (free, no key)
  - Notifications: ntfy.sh (free push notification app, no account needed)
  - Hosting/scheduling: GitHub Actions free tier (see SETUP below) --
    a free GitHub account gives you free scheduled runs, no server rental.
Nothing in here requires a credit card.

SETUP (all free)
-----------------
1. Install the ntfy app (iOS/Android) or use ntfy.sh in a browser.
   Pick a random, hard-to-guess topic name (it's like a public inbox),
   e.g. "nse-picks-x7q2p9". Subscribe to it in the app. This is your
   notification channel -- totally free, no signup.
2. Put that topic name into NTFY_TOPIC in the CONFIG section below.
3. Test locally first (optional but recommended):
     pip install -r requirements.txt
     python nse_trading_assistant.py --run-now
4. Run it daily for free using GitHub Actions (no server, no bill):
     - Create a free GitHub account if you don't have one.
     - Make a new repo, upload this file + requirements.txt.
     - Add the workflow file from daily.yml (provided alongside this
       script) into .github/workflows/daily.yml in that repo.
     - GitHub will now run this script automatically every weekday
       morning, for free, and push the result straight to your phone
       via ntfy. You don't need Emergent, a VPS, or any paid hosting
       for this version.
"""

import os
import sys
import json
import time
import csv
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# =============================================================================
# CONFIG -- edit these
# =============================================================================

# Your watchlist. Use NSE ticker + ".NS" suffix.
# This is a set of the largest, most liquid Nifty 50 stocks by free-float
# market cap (verified against current index weightings) -- deliberately
# avoids thinly-traded small-caps, since low liquidity produces noisy
# price swings that look like "signals" but are often just illiquidity.
WATCHLIST = [
    "RELIANCE.NS", "BHARTIARTL.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
    "TCS.NS", "BAJFINANCE.NS", "LT.NS", "HINDUNILVR.NS", "TITAN.NS",
    "SUNPHARMA.NS", "INFY.NS", "KOTAKBANK.NS", "AXISBANK.NS", "MARUTI.NS",
    "ULTRACEMCO.NS", "ITC.NS",
]

# ntfy.sh topic -- pick something private/hard to guess, subscribe to it
# in the ntfy app. No account needed.
NTFY_TOPIC = "sam-nse-alerts-4471"
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"

# Minimum composite score (0-100) required before the script will produce
# a pick at all. Raise this if you want fewer, higher-conviction signals.
MIN_SCORE_TO_TRADE = 65

# Buy/sell time window (IST). NSE opens 9:15. We wait a bit for the open
# volatility to settle before suggesting entry, and force an exit well
# before the 3:30 close to avoid closing-auction chaos.
BUY_WINDOW_START = "09:20"
BUY_WINDOW_END = "09:45"
SELL_BY_TIME = "15:00"

LOG_FILE = Path(__file__).parent / "picks_log.csv"

DISCLAIMER = (
    "This is an automated research signal, not financial advice, and the "
    "target return is a projection based on volatility/technicals -- not "
    "a guarantee. Size your position according to your own risk tolerance."
)

# =============================================================================
# TECHNICAL ANALYSIS
# =============================================================================

def fetch_history(symbol: str, period: str = "6mo") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(symbol).history(period=period, interval="1d")
        if df.empty or len(df) < 30:
            return None
        return df
    except Exception as e:
        print(f"[warn] failed to fetch {symbol}: {e}")
        return None


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def technical_signal(symbol: str) -> dict | None:
    df = fetch_history(symbol)
    if df is None:
        return None

    close = df["Close"]
    rsi = compute_rsi(close).iloc[-1]
    macd_line, signal_line, hist = compute_macd(close)
    macd_bullish = hist.iloc[-1] > 0 and hist.iloc[-2] <= 0  # fresh crossover
    atr = compute_atr(df).iloc[-1]
    last_price = close.iloc[-1]
    avg_volume_20 = df["Volume"].tail(20).mean()
    last_volume = df["Volume"].iloc[-1]
    volume_spike_ratio = last_volume / avg_volume_20 if avg_volume_20 else 1.0

    recent_high = df["High"].tail(20).max()
    recent_low = df["Low"].tail(20).min()
    dist_from_resistance_pct = (recent_high - last_price) / last_price * 100
    dist_from_support_pct = (last_price - recent_low) / last_price * 100

    # --- simple composite scoring (0-100), tune freely ---
    score = 50.0
    if 45 <= rsi <= 65:            # healthy momentum, not overbought/oversold
        score += 10
    elif rsi > 70:
        score -= 10                # overbought, risky to chase
    elif rsi < 30:
        score -= 5                 # oversold, could be falling knife

    if macd_bullish:
        score += 15

    if volume_spike_ratio >= 1.5:
        score += 15
    elif volume_spike_ratio >= 1.2:
        score += 7

    if dist_from_resistance_pct < 2:
        score -= 8                 # right under resistance, riskier breakout
    if dist_from_support_pct < 2:
        score += 5                 # bouncing off support, decent entry

    score = max(0.0, min(100.0, score))

    return {
        "symbol": symbol,
        "last_price": round(float(last_price), 2),
        "rsi": round(float(rsi), 1) if not np.isnan(rsi) else None,
        "macd_bullish_cross": bool(macd_bullish),
        "atr": round(float(atr), 2) if not np.isnan(atr) else None,
        "volume_spike_ratio": round(float(volume_spike_ratio), 2),
        "recent_high_20d": round(float(recent_high), 2),
        "recent_low_20d": round(float(recent_low), 2),
        "technical_score": round(score, 1),
    }


# =============================================================================
# NEWS SENTIMENT -- FREE VERSION (Google News RSS, no API key, no cost)
# =============================================================================
#
# This pulls recent headlines for free via Google News' public RSS search
# (no account, no key, no billing) and scores sentiment with a simple
# keyword lexicon. It's cruder than an LLM read of the news -- it can't
# tell sarcasm from sincerity or weigh context -- but it costs nothing and
# is enough to nudge the technical score, not replace it. If you ever want
# sharper news reads later, this function is the only place you'd swap in
# a paid option.

import re
import xml.etree.ElementTree as ET
from urllib.parse import quote

POSITIVE_WORDS = {
    "surge", "surges", "jumps", "jump", "rally", "rallies", "soar", "soars",
    "gain", "gains", "beat", "beats", "record", "high", "upgrade", "upgraded",
    "profit", "profits", "growth", "expands", "expansion", "strong", "boost",
    "boosts", "wins", "win", "positive", "bullish", "outperform", "rises", "rise",
}
NEGATIVE_WORDS = {
    "falls", "fall", "plunge", "plunges", "drop", "drops", "crash", "crashes",
    "downgrade", "downgraded", "loss", "losses", "miss", "misses", "weak",
    "decline", "declines", "probe", "fraud", "lawsuit", "penalty", "fine",
    "negative", "bearish", "underperform", "slump", "slumps", "cuts", "cut",
    "layoffs", "resigns", "scam",
}


def get_news_sentiment(symbol: str) -> dict:
    """
    Free news sentiment: fetches recent headlines from Google News RSS
    (public, no key needed) and does simple keyword counting.
    Fails soft to neutral on any error, so a single stock's news hiccup
    never crashes the whole run.
    """
    company = symbol.replace(".NS", "")
    query = quote(f"{company} NSE stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        titles = [item.findtext("title") or "" for item in root.findall(".//item")][:10]

        if not titles:
            return {"sentiment_score": 0, "reason": "no recent news found"}

        pos_hits, neg_hits = 0, 0
        for title in titles:
            words = set(re.findall(r"[a-zA-Z]+", title.lower()))
            pos_hits += len(words & POSITIVE_WORDS)
            neg_hits += len(words & NEGATIVE_WORDS)

        raw_score = (pos_hits - neg_hits) * 4  # scale up, then clamp
        score = max(-20, min(20, raw_score))

        if score > 5:
            reason = f"news tone leans positive ({pos_hits} positive vs {neg_hits} negative signal words)"
        elif score < -5:
            reason = f"news tone leans negative ({neg_hits} negative vs {pos_hits} positive signal words)"
        else:
            reason = "news tone roughly neutral / mixed"

        return {"sentiment_score": score, "reason": reason}
    except Exception as e:
        print(f"[warn] news sentiment failed for {symbol}: {e}")
        return {"sentiment_score": 0, "reason": "news check failed, treated as neutral"}


# =============================================================================
# SIGNAL BUILDING
# =============================================================================

def build_signal(tech: dict, news: dict) -> dict:
    composite_score = max(0.0, min(100.0, tech["technical_score"] + news["sentiment_score"]))
    last_price = tech["last_price"]
    atr = tech["atr"] or (last_price * 0.015)  # fallback ~1.5% if ATR missing

    # Target/stop-loss sized off ATR (a standard, honest way to size moves
    # to a stock's OWN typical volatility, instead of an arbitrary fixed %).
    target_price = round(last_price + 1.5 * atr, 2)
    stop_loss_price = round(last_price - 1.0 * atr, 2)
    projected_return_pct = round((target_price - last_price) / last_price * 100, 2)
    risk_pct = round((last_price - stop_loss_price) / last_price * 100, 2)

    buy_zone_low = 
