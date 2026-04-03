import os
import re
import json
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import requests
from dateutil.relativedelta import relativedelta
from typing import TypedDict
from dotenv import load_dotenv
from sklearn.covariance import GraphicalLasso
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent as create_react_agent

load_dotenv()

# ==========================================
# 1. STATE
# ==========================================

class PortfolioState(TypedDict):
    tickers: list[str]
    target_date: str
    fundamental_analysis: dict[str, str]
    technical_analysis: dict[str, dict]
    quant_analysis: str
    sentiment_analysis: dict[str, float]
    risk_analysis: dict[str, dict]
    final_decision: str
    trade_book: dict

# ==========================================
# 2. TOOLS
# ==========================================

@tool
def get_fundamentals(ticker: str) -> str:
    """Fetch fundamental metrics from Finnhub"""
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return "API key missing"

    base = "https://finnhub.io/api/v1"
    params = {"symbol": ticker.upper(), "token": api_key}

    profile = requests.get(f"{base}/stock/profile2", params=params).json()
    metrics = requests.get(
        f"{base}/stock/metric",
        params={**params, "metric": "all"}
    ).json().get("metric", {})

    pe = metrics.get("peTTM", "N/A")
    debt = metrics.get("totalDebt/totalEquityAnnual", "N/A")

    return f"{ticker} | P/E:{pe} | Debt/Equity:{debt}"

@tool
def get_technicals(ticker: str, target_date: str) -> str:
    """Compute RSI, MACD and SMA200"""
    end = datetime.datetime.strptime(target_date, "%Y-%m-%d")
    start = end - relativedelta(months=9)

    df = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d")
    )

    if df.empty:
        return f"{ticker} | No data"

    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.sma(length=200, append=True)

    last = df.iloc[-1]
    trend = "Above SMA200" if last["Close"] > last.get("SMA_200", last["Close"]) else "Below SMA200"

    return {
    "ticker": ticker,
    "price": float(last["Close"]),
    "rsi": float(last.get("RSI_14", 50)),
    "macd": float(last.get("MACD_12_26_9", 0)),
    "signal": float(last.get("MACDs_12_26_9", 0)),
    "trend": trend
}



def calculate_markowitz_long_short(tickers: list[str], target_date: str) -> str:
    """Mean-Variance portfolio optimization"""
    end = datetime.datetime.strptime(target_date,"%Y-%m-%d")
    start = end - relativedelta(years=2)

    prices = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False
    )["Close"].dropna(axis=1)

    returns = prices.pct_change().dropna()
    mu = returns.mean().values
    centered = returns - returns.mean()
    
    glasso = GraphicalLasso(alpha=0.05).fit(centered)
    precision = glasso.precision_
    
    raw_weights = precision @ mu
    weights = raw_weights / np.sum(np.abs(raw_weights))

    return "\n".join([f"{t}: {w:.3f}" for t, w in zip(prices.columns, weights)])

@tool
def calculate_sl_tp(ticker: str) -> str:
    """ATR-based stop loss and take profit"""
    data = yf.download(ticker, period="1mo", progress=False)
    if data.empty:
        return f"{ticker} | No data"

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(14).mean().iloc[-1]
    price = close.iloc[-1]
    sl = price - 1.5 * atr
    tp = price + 3 * atr

    return f"{ticker} | Price:{price:.2f} SL:{sl:.2f} TP:{tp:.2f}"

def adjust_by_tech_and_sentiment(price, atr, tech_state, sentiment):
    sl_factor = 1.0
    tp_factor = 1.0

    if sentiment > 0.5:
        sl_factor *= 1.1
        tp_factor *= 1.2
    elif sentiment < -0.5:
        sl_factor *= 0.8
        tp_factor *= 0.8

    if tech_state.get("trend") == "Below SMA200":
        sl_factor *= 0.8
        tp_factor *= 0.9

    rsi_val = tech_state.get("rsi", 50)

    if rsi_val > 70:
        sl_factor *= 0.8
    elif rsi_val < 30:
        sl_factor *= 1.2

    base_sl = 1.5 * atr
    base_tp = 3.0 * atr

    return {
        "price": float(price),
        "sl": round(price - base_sl * sl_factor, 2),
        "tp": round(price + base_tp * tp_factor, 2),
        "sl_factor": round(sl_factor, 2),
        "tp_factor": round(tp_factor, 2)
    }
# ==========================================
# 3. AGENTS
# ==========================================

llm = ChatOllama(model="phi3", temperature=0.1)
risk_agent = create_react_agent(
    llm,
    tools=[calculate_sl_tp],
    system_prompt="You are a risk manager. Compute SL and TP using ATR."
)

# ==========================================
# 4. GRAPH NODES
# ==========================================

def node_fundamentals(state: PortfolioState):
    results = {}

    for ticker in state["tickers"]:
        try:
            data = get_fundamentals.invoke({"ticker": ticker})
            results[ticker] = data
        except Exception as e:
            results[ticker] = f"Error: {str(e)}"

    return {"fundamental_analysis": results}

def node_technicals(state: PortfolioState):
    results = {}

    for ticker in state["tickers"]:
        try:
            data = get_technicals.invoke({
                "ticker": ticker,
                "target_date": state["target_date"]
            })
            results[ticker] = data
        except Exception as e:
            results[ticker] = {"error": str(e)}

    return {"technical_analysis": results}

def node_sentiment(state: PortfolioState):
    """Compute sentiment scores directly in the node."""
    sentiment_scores = {}

    for ticker in state["tickers"]:
        try:
            news = yf.Ticker(ticker).news or []
            headlines = [n.get("title", "") for n in news[:5]]

            if not headlines:
                sentiment_scores[ticker] = 0.0
                continue

            # Build prompt for LLM
            prompt = f"""
            Evaluate the sentiment of these headlines for the stock {ticker}.
            Return ONLY a number between -1 (very negative) and +1 (very positive).

            Headlines:
            {headlines}
            """

            score_str = llm.invoke([{"role":"user","content":prompt}]).content.strip()
            try:
                score = float(score_str)
            except ValueError:
                score = 0.0  # fallback
            sentiment_scores[ticker] = max(min(score, 1), -1)  # clamp between -1 and 1

        except Exception as e:
            sentiment_scores[ticker] = 0.0

    # Store in state
    return {"sentiment_analysis": sentiment_scores}
def node_quant(state: PortfolioState):
    try:
        result = calculate_markowitz_long_short(
            tickers=state["tickers"],
            target_date=state["target_date"]
        )
        return {"quant_analysis": result}
    except Exception as e:
        return {"quant_analysis": f"Error: {str(e)}"}

def node_risk(state: PortfolioState):
    risk_results = {}
    tech_data = state.get("technical_analysis", {})
    sentiment_data = state.get("sentiment_analysis", {})

    for ticker in state.get("tickers", []):
        try:
            data = yf.download(ticker, period="3mo", progress=False)
            if data.empty or len(data) < 15:
                risk_results[ticker] = {"error": "Insufficient data"}
                continue

            # FIX : aplatir le multi-index si présent
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            high  = data["High"]
            low   = data["Low"]
            close = data["Close"]

            tr = pd.concat([
                high - low,
                abs(high - close.shift()),
                abs(low  - close.shift())
            ], axis=1).max(axis=1)

            # FIX : forcer les scalaires avec float()
            atr   = float(tr.rolling(14).mean().iloc[-1])
            price = float(close.iloc[-1])

            sentiment_score = sentiment_data.get(ticker, 0.0)
            tech_state      = tech_data.get(ticker, {})

            risk_profile = adjust_by_tech_and_sentiment(
                price=price, atr=atr,
                tech_state=tech_state, sentiment=sentiment_score
            )
            risk_results[ticker] = risk_profile

        except Exception as e:
            risk_results[ticker] = {"error": str(e)}

    return {"risk_analysis": risk_results}

def node_trader(state: PortfolioState):
    lines = []
    for ticker in state["tickers"]:
        fund = state["fundamental_analysis"].get(ticker, "N/A")
        tech = state["technical_analysis"].get(ticker, {})
        sent = state["sentiment_analysis"].get(ticker, 0.0)
        risk = state["risk_analysis"].get(ticker, {})
        lines.append(
            f"- {ticker}: {fund} | RSI={tech.get('rsi','?')} "
            f"trend={tech.get('trend','?')} | sentiment={sent:.2f} | "
            f"price={risk.get('price','?')} SL={risk.get('sl','?')} TP={risk.get('tp','?')}"
        )

    summary = "\n".join(lines)
    quant   = state["quant_analysis"]

    prompt = f"""You are a portfolio manager. Analyze these stocks:

{summary}

Quant weights:
{quant}

You MUST reply with ONLY these lines, one per stock, no extra text:
{chr(10).join([f"{t} | BUY | 0.2 | reason here" for t in state["tickers"]])}

Replace BUY with BUY, SELL or SKIP. Replace 0.2 with a number. Replace reason here with one short reason.
Do not add any other text before or after."""

    response = llm.invoke([{"role": "user", "content": prompt}])
    raw = response.content.strip()
    print(f"\n[Trader raw output]\n{raw}\n")

    trade_book      = {}
    rationale_lines = []
    tickers_upper   = [t.upper() for t in state["tickers"]]

    for line in raw.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]

        ticker_candidate = parts[0].upper().lstrip("-").strip()

        if ticker_candidate not in tickers_upper:
            continue

        
        action = "SKIP"
        for word in ["BUY", "SELL", "SKIP"]:
            if any(word in p.upper() for p in parts):
                action = word
                break

        # Extraire le poids (premier float trouvé)
        weight = 0.0
        for p in parts:
            try:
                weight = float(p.strip())
                weight = max(-1.0, min(1.0, weight))
                break
            except ValueError:
                continue

        reason = parts[-1] if len(parts) >= 4 else ""
        trade_book[ticker_candidate]  = {"action": action, "weight": weight}
        rationale_lines.append(f"{ticker_candidate}: {action} ({weight:+.2f}) — {reason}")

    # Fallback si rien parsé du tout
    if not trade_book:
        print("[Trader] Parsing failed, applying SKIP to all tickers")
        trade_book      = {t: {"action": "SKIP", "weight": 0.0} for t in state["tickers"]}
        rationale_lines = ["Could not parse trader output"]

    return {
        "final_decision": "\n".join(rationale_lines),
        "trade_book":     trade_book
    }
# ==========================================
# 5. GRAPH
# ==========================================


graph = StateGraph(PortfolioState)

graph.add_node("fundamentals", node_fundamentals)
graph.add_node("technicals", node_technicals)
graph.add_node("sentiment", node_sentiment)
graph.add_node("quant", node_quant)
graph.add_node("risk", node_risk)
graph.add_node("trader", node_trader)

graph.add_edge(START, "fundamentals")
graph.add_edge(START, "technicals")
graph.add_edge(START, "sentiment")
graph.add_edge(START, "quant")
graph.add_edge(["technicals", "sentiment"], "risk")
graph.add_edge("fundamentals", "trader")
graph.add_edge("quant", "trader")
graph.add_edge("sentiment", "trader")  
graph.add_edge("risk", "trader")
graph.add_edge("trader", END)

app = graph.compile()




