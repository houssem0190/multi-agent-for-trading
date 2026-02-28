import os
import re
import json
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import requests
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from typing import TypedDict
from dotenv import load_dotenv
from sklearn.covariance import GraphicalLasso
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent as create_react_agent

load_dotenv()

# =============================================================================
# 1. STATE
#    No market_direction — the trader owns that decision entirely.
# =============================================================================
class PortfolioState(TypedDict):
    tickers:              list[str]
    target_date:          str
    fundamental_analysis: str
    technical_analysis:   str
    sentiment_analysis:   str
    quant_analysis:       str   # GLASSO weights (long AND short) + macro
    risk_analysis:        str   # risk agent synthesis
    final_decision:       str   # trader rationale string
    trade_book:           dict  # parsed {ticker: {action, weight, reason}}


# =============================================================================
# 2. TOOLS
# =============================================================================

@tool
def get_fundamentals(ticker: str) -> str:
    """
    Fetches company fundamentals from Finnhub:
    industry, market cap, P/E TTM, EV/EBITDA, debt/equity, ROE, revenue growth.
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return "FINNHUB_API_KEY missing."

    symbol = ticker.strip().upper()
    base   = "https://finnhub.io/api/v1"
    p      = {"symbol": symbol, "token": api_key}

    try:
        profile = requests.get(f"{base}/stock/profile2", params=p, timeout=15).json() or {}
        m = (
            requests.get(f"{base}/stock/metric", params={**p, "metric": "all"}, timeout=15).json() or {}
        ).get("metric", {})
    except requests.RequestException as e:
        return f"Finnhub error for {symbol}: {e}"

    return (
        f"{profile.get('name', symbol)} ({symbol}) | "
        f"Industry: {profile.get('finnhubIndustry','N/A')} | "
        f"MktCap: {profile.get('marketCapitalization','N/A')}M | "
        f"P/E TTM: {m.get('peTTM','N/A')} | "
        f"EV/EBITDA: {m.get('evEbitdaTTM','N/A')} | "
        f"Debt/Eq: {m.get('totalDebt/totalEquityAnnual','N/A')} | "
        f"ROE: {m.get('roeTTM','N/A')}% | "
        f"Rev growth (YoY): {m.get('revenueGrowthTTMYoy','N/A')}%"
    )


@tool
def get_technicals(ticker: str, target_date: str) -> str:
    """
    Calculates RSI-14, MACD, SMA-50, SMA-200, Bollinger Bands, ATR-14,
    and 1-month / 3-month momentum — all point-in-time at target_date.
    """
    end   = datetime.datetime.strptime(target_date, "%Y-%m-%d")
    start = end - relativedelta(months=9)

    df = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
    )
    if df.empty:
        return f"No price data for {ticker}."

    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.sma(length=50,  append=True)
    df.ta.sma(length=200, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)

    last  = df.iloc[-1]
    close = float(last["Close"])

    rsi    = last.get("RSI_14", 50)
    macd_v = last.get("MACD_12_26_9", 0)
    macd_s = last.get("MACDs_12_26_9", 0)
    sma50  = last.get("SMA_50",  close)
    sma200 = last.get("SMA_200", close)
    bbu    = last.get("BBU_20_2.0", close)
    bbl    = last.get("BBL_20_2.0", close)
    atr    = last.get("ATRr_14", 0)

    mom1m = ((close - float(df["Close"].iloc[-22])) / float(df["Close"].iloc[-22]) * 100) if len(df) > 22 else 0
    mom3m = ((close - float(df["Close"].iloc[-63])) / float(df["Close"].iloc[-63]) * 100) if len(df) > 63 else 0

    return (
        f"{ticker} @ {target_date} | Price: {close:.2f} | "
        f"RSI: {float(rsi):.1f} ({'Overbought' if float(rsi)>70 else 'Oversold' if float(rsi)<30 else 'Neutral'}) | "
        f"MACD: {float(macd_v):.3f} vs Signal: {float(macd_s):.3f} ({'Bull' if float(macd_v)>float(macd_s) else 'Bear'} cross) | "
        f"SMA50: {float(sma50):.2f} SMA200: {float(sma200):.2f} ({'Above' if close>float(sma200) else 'Below'} 200d) | "
        f"BB: [{float(bbl):.2f} – {float(bbu):.2f}] | ATR: {float(atr):.3f} | "
        f"Mom 1M: {mom1m:+.1f}% | Mom 3M: {mom3m:+.1f}%"
    )


@tool
def get_sentiment(ticker: str) -> str:
    """
    Fetches recent news headlines + Finnhub insider sentiment to gauge market mood.
    """
    api_key = os.getenv("FINNHUB_API_KEY")

    try:
        news  = yf.Ticker(ticker).news or []
        lines = [f"- {n.get('title','?')}" for n in news[:7]]
        report = f"Headlines for {ticker}:\n" + "\n".join(lines)
    except Exception:
        report = f"Could not fetch news for {ticker}."

    if api_key:
        try:
            today   = datetime.date.today()
            three_m = (today - relativedelta(months=3)).strftime("%Y-%m-%d")
            resp = requests.get(
                "https://finnhub.io/api/v1/stock/insider-sentiment",
                params={"symbol": ticker, "from": three_m, "to": str(today), "token": api_key},
                timeout=10,
            ).json()
            data = resp.get("data", [])
            if data:
                latest = data[-1]
                report += (
                    f"\nInsider Sentiment: "
                    f"MSPR={latest.get('mspr','N/A')} | Change={latest.get('change','N/A')}"
                )
        except Exception:
            pass

    return report


@tool
def calculate_markowitz_long_short(tickers: list[str], target_date: str) -> str:
    """
    GLASSO precision-matrix optimization that ALLOWS negative weights (shorts).
    Long book sums to +100%, short book is capped at -30%.
    Returns LONG / SHORT / SKIP label and weight % per ticker.
    """
    if len(tickers) < 2:
        return "Need at least 2 tickers."

    end   = datetime.datetime.strptime(target_date, "%Y-%m-%d")
    start = end - relativedelta(years=2)

    try:
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
        )["Close"].dropna(axis=1, how="all")

        valid_tickers = list(raw.columns)
        returns       = raw.pct_change().dropna()
        centered      = returns - returns.mean()

        glasso = GraphicalLasso(alpha=0.5, max_iter=200)
        glasso.fit(centered)
        prec = glasso.precision_

        ones        = np.ones(len(valid_tickers))
        raw_weights = prec @ ones / (ones @ prec @ ones)

        longs  = np.where(raw_weights > 0, raw_weights, 0.0)
        shorts = np.where(raw_weights < 0, raw_weights, 0.0)

        l_sum = longs.sum()
        s_sum = shorts.sum()

        if l_sum > 0:
            longs = longs / l_sum
        if s_sum < 0:
            shorts = shorts / abs(s_sum) * 0.30

        final_weights = longs + shorts

        lines = [f"GLASSO Long-Short Weights as of {target_date} (2yr, alpha=0.5):"]
        for t, w in sorted(zip(valid_tickers, final_weights), key=lambda x: -x[1]):
            if w > 0.001:
                label = "LONG "
            elif w < -0.001:
                label = "SHORT"
            else:
                label = "SKIP "
            lines.append(f"  {label} {t:<6}: {w*100:>+7.2f}%")

        return "\n".join(lines)

    except Exception as e:
        return f"GLASSO failed: {e}"


@tool
def get_macro_context(target_date: str) -> str:
    """
    Point-in-time macro snapshot: SPY trend, VIX, DXY, TLT, and sector ETFs.
    """
    end   = datetime.datetime.strptime(target_date, "%Y-%m-%d")
    start = end - relativedelta(months=3)
    s, e  = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    proxies = {
        "SPY":      "S&P 500",
        "^VIX":     "VIX Fear Index",
        "DX-Y.NYB": "US Dollar (DXY)",
        "TLT":      "20Y Bonds",
        "XLK":      "Tech sector",
        "XLF":      "Financials",
        "XLE":      "Energy",
        "XLV":      "Healthcare",
        "XLU":      "Utilities (defensive)",
    }

    lines = [f"Macro snapshot as of {target_date}:"]
    for sym, label in proxies.items():
        try:
            px = yf.download(sym, start=s, end=e, progress=False)["Close"].dropna()
            if len(px) < 5:
                continue
            current   = float(px.iloc[-1])
            one_m_ago = float(px.iloc[-22]) if len(px) >= 22 else float(px.iloc[0])
            chg       = (current - one_m_ago) / one_m_ago * 100
            trend     = "↑" if chg > 0 else "↓"
            lines.append(f"  {label:<30} ({sym:<10}): {current:>9.2f}  {trend} {chg:>+5.1f}% (1M)")
        except Exception:
            pass

    return "\n".join(lines)


# =============================================================================
# 3. LLM + AGENTS
# =============================================================================
llm = ChatOllama(model="phi3", temperature=0.1)

fund_agent  = create_react_agent(llm, tools=[get_fundamentals],
    system_prompt="You are a senior fundamental analyst. Be concise and data-driven.")
tech_agent  = create_react_agent(llm, tools=[get_technicals],
    system_prompt="You are a senior technical analyst. Summarize signals clearly per ticker.")
sent_agent  = create_react_agent(llm, tools=[get_sentiment],
    system_prompt="You are a market sentiment analyst. Assess news tone and insider flows.")
quant_agent = create_react_agent(
    llm,
    tools=[calculate_markowitz_long_short, get_macro_context],
    system_prompt=(
        "You are a quantitative analyst. "
        "Step 1: fetch macro context. "
        "Step 2: run GLASSO long-short optimization. "
        "Summarize macro regime and per-ticker LONG / SHORT / SKIP signals."
    ),
)


def _is_tool_error(exc: Exception) -> bool:
    return "does not support tools" in str(exc).lower()


# =============================================================================
# 4. GRAPH NODES
# =============================================================================

def node_fundamentals(state: PortfolioState):
    prompt = (
        f"Today is {state['target_date']}. "
        f"Analyze fundamentals for each ticker: {state['tickers']}. "
        "Call the tool for each and give a 1-2 sentence verdict per ticker."
    )
    try:
        r = fund_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"fundamental_analysis": r["messages"][-1].content}
    except Exception as exc:
        if not _is_tool_error(exc):
            raise
        reports = [get_fundamentals.invoke({"ticker": t}) for t in state["tickers"]]
        return {"fundamental_analysis": "\n".join(reports)}


def node_technicals(state: PortfolioState):
    prompt = (
        f"Today is {state['target_date']}. "
        f"Analyze technicals for: {state['tickers']} — always pass target_date='{state['target_date']}'. "
        "Summarize momentum, trend and overbought/oversold per ticker."
    )
    try:
        r = tech_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"technical_analysis": r["messages"][-1].content}
    except Exception as exc:
        if not _is_tool_error(exc):
            raise
        reports = [
            get_technicals.invoke({"ticker": t, "target_date": state["target_date"]})
            for t in state["tickers"]
        ]
        return {"technical_analysis": "\n".join(reports)}


def node_sentiment(state: PortfolioState):
    prompt = (
        f"Today is {state['target_date']}. "
        f"Analyze news sentiment and insider flows for: {state['tickers']}. "
        "Give bullish / neutral / bearish verdict per ticker."
    )
    try:
        r = sent_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"sentiment_analysis": r["messages"][-1].content}
    except Exception as exc:
        if not _is_tool_error(exc):
            raise
        reports = [get_sentiment.invoke({"ticker": t}) for t in state["tickers"]]
        return {"sentiment_analysis": "\n".join(reports)}


def node_quant(state: PortfolioState):
    prompt = (
        f"Today is {state['target_date']}. "
        f"Step 1 — call get_macro_context with target_date='{state['target_date']}'. "
        f"Step 2 — call calculate_markowitz_long_short with "
        f"tickers={state['tickers']} and target_date='{state['target_date']}'. "
        "Summarize macro regime and per-ticker quant signal."
    )
    try:
        r = quant_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"quant_analysis": r["messages"][-1].content}
    except Exception as exc:
        if not _is_tool_error(exc):
            raise
        macro   = get_macro_context.invoke({"target_date": state["target_date"]})
        weights = calculate_markowitz_long_short.invoke(
            {"tickers": state["tickers"], "target_date": state["target_date"]}
        )
        return {"quant_analysis": macro + "\n\n" + weights}


def node_risk(state: PortfolioState):
    """
    Synthesizes all 4 analyst reports into a per-ticker conviction score
    (-3 strong short to +3 strong long) and overall risk level.
    No routing — just an honest read for the trader.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Chief Risk Officer. Synthesize 4 analyst reports. "
                "For EACH ticker output a conviction score -3 (strong short) to +3 (strong long) "
                "with one sentence of reasoning. "
                "Also state overall portfolio risk: LOW / MEDIUM / HIGH. "
                "Be objective — do not default to bullish."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Date: {state['target_date']}\nTickers: {state['tickers']}\n\n"
                f"=== FUNDAMENTALS ===\n{state.get('fundamental_analysis','N/A')}\n\n"
                f"=== TECHNICALS ===\n{state.get('technical_analysis','N/A')}\n\n"
                f"=== SENTIMENT ===\n{state.get('sentiment_analysis','N/A')}\n\n"
                f"=== QUANT / MACRO ===\n{state.get('quant_analysis','N/A')}\n\n"
                "Provide per-ticker scores and overall risk level."
            ),
        },
    ]
    r = llm.invoke(messages)
    return {"risk_analysis": r.content}


def node_trader(state: PortfolioState):
    """
    Reads all 5 reports. Decides LONG, SHORT, or SKIP per ticker independently.
    Outputs a structured JSON trade book — no regime flag needed.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Head Trader. You have full autonomy: LONG, SHORT, or SKIP any ticker. "
                "Output ONLY a JSON object — no markdown, no explanation outside the JSON.\n"
                "Format:\n"
                '{"rationale": "one paragraph", "trades": {'
                '"TICKER": {"action": "LONG|SHORT|SKIP", "weight": 0.00, "reason": "short reason"}'
                ", ...}}\n"
                "Weight rules:\n"
                "- Decimals (0.15 = 15%). Negative for shorts (e.g. -0.10).\n"
                "- LONG weights must sum between 0.80 and 1.00.\n"
                "- SHORT weights must sum between -0.30 and 0.00.\n"
                "- SKIP entries must have weight 0.0.\n"
                "- HIGH risk regime: reduce longs, increase shorts.\n"
                "- Use GLASSO weights as starting point, adjust based on all reports."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Date: {state['target_date']}\nTickers: {', '.join(state['tickers'])}\n\n"
                f"=== FUNDAMENTALS ===\n{state.get('fundamental_analysis','N/A')}\n\n"
                f"=== TECHNICALS ===\n{state.get('technical_analysis','N/A')}\n\n"
                f"=== SENTIMENT ===\n{state.get('sentiment_analysis','N/A')}\n\n"
                f"=== QUANT / MACRO ===\n{state.get('quant_analysis','N/A')}\n\n"
                f"=== RISK SYNTHESIS ===\n{state.get('risk_analysis','N/A')}\n\n"
                "Output ONLY the JSON trade book."
            ),
        },
    ]

    r     = llm.invoke(messages)
    raw   = r.content.strip()
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    trade_book: dict = {}
    rationale: str   = raw

    try:
        parsed    = json.loads(clean)
        trades    = parsed.get("trades", {})
        rationale = parsed.get("rationale", raw)

        # Rescale if model blew the weight constraints
        long_total  = sum(v["weight"] for v in trades.values() if v.get("weight", 0) > 0)
        short_total = sum(v["weight"] for v in trades.values() if v.get("weight", 0) < 0)

        if long_total > 1.05:
            scale = 1.0 / long_total
            for k in trades:
                if trades[k]["weight"] > 0:
                    trades[k]["weight"] = round(trades[k]["weight"] * scale, 4)

        if short_total < -0.35:
            scale = 0.30 / abs(short_total)
            for k in trades:
                if trades[k]["weight"] < 0:
                    trades[k]["weight"] = round(trades[k]["weight"] * scale, 4)

        trade_book = trades

    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: equal-weight long everything
        eq_w = round(1.0 / len(state["tickers"]), 4)
        trade_book = {
            t: {"action": "LONG", "weight": eq_w, "reason": "JSON parse fallback — equal weight"}
            for t in state["tickers"]
        }

    return {"final_decision": rationale, "trade_book": trade_book}


# =============================================================================
# 5. BUILD GRAPH
#    4 analysts fire in parallel → risk synthesizer → trader
#    No conditional routing. The trader decides everything.
# =============================================================================
graph = StateGraph(PortfolioState)

graph.add_node("fundamentals", node_fundamentals)
graph.add_node("technicals",   node_technicals)
graph.add_node("sentiment",    node_sentiment)
graph.add_node("quant",        node_quant)
graph.add_node("risk",         node_risk)
graph.add_node("trader",       node_trader)

graph.add_edge(START, "fundamentals")
graph.add_edge(START, "technicals")
graph.add_edge(START, "sentiment")
graph.add_edge(START, "quant")

graph.add_edge(["fundamentals", "technicals", "sentiment", "quant"], "risk")
graph.add_edge("risk",   "trader")
graph.add_edge("trader", END)

app = graph.compile()
from IPython.display import Image, display

display(Image(app.get_graph().draw_png()))

