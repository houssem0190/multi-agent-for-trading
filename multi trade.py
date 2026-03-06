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
    fundamental_analysis: str
    technical_analysis: str
    sentiment_analysis: str
    quant_analysis: str
    risk_analysis: str
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

    return f"""{ticker}
Price:{last['Close']:.2f}
RSI:{last.get('RSI_14',50):.1f}
MACD:{last.get('MACD_12_26_9',0):.3f}
Signal:{last.get('MACDs_12_26_9',0):.3f}
Trend:{trend}"""

@tool
def get_sentiment(ticker: str) -> str:
    """Fetch recent news headlines"""
    try:
        news = yf.Ticker(ticker).news or []
        headlines = [n.get("title","?") for n in news[:5]]
        return f"{ticker} Headlines:\n" + "\n".join(headlines)
    except:
        return "No news available"

@tool
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

# ==========================================
# 3. AGENTS
# ==========================================

llm = ChatOllama(model="phi3", temperature=0.1)

fund_agent = create_react_agent(
    llm,
    tools=[get_fundamentals],
    system_prompt="You are a fundamental analyst. Call the tool for each ticker and summarize."
)

tech_agent = create_react_agent(
    llm,
    tools=[get_technicals],
    system_prompt="You are a technical analyst. Evaluate RSI, MACD and trend."
)

sent_agent = create_react_agent(
    llm,
    tools=[get_sentiment],
    system_prompt="You are a sentiment analyst. Evaluate market tone."
)

quant_agent = create_react_agent(
    llm,
    tools=[calculate_markowitz_long_short],
    system_prompt="You are a quantitative portfolio optimizer."
)

risk_agent = create_react_agent(
    llm,
    tools=[calculate_sl_tp],
    system_prompt="You are a risk manager. Compute SL and TP using ATR."
)

# ==========================================
# 4. GRAPH NODES
# ==========================================

def node_fundamentals(state: PortfolioState):
    prompt = f"Analyze fundamentals for these tickers.\nTickers:\n{state['tickers']}"
    r = fund_agent.invoke({"messages":[{"role":"user","content":prompt}]})
    return {"fundamental_analysis": r["messages"][-1].content}

def node_technicals(state: PortfolioState):
    prompt = f"Analyze technical signals for each ticker.\nTickers:{state['tickers']}\nDate:{state['target_date']}"
    r = tech_agent.invoke({"messages":[{"role":"user","content":prompt}]})
    return {"technical_analysis": r["messages"][-1].content}

def node_sentiment(state: PortfolioState):
    prompt = f"Analyze sentiment and news for each ticker.\nTickers:{state['tickers']}"
    r = sent_agent.invoke({"messages":[{"role":"user","content":prompt}]})
    return {"sentiment_analysis": r["messages"][-1].content}

def node_quant(state: PortfolioState):
    prompt = f"Run portfolio optimization on those assets.\nTickers:{state['tickers']}\nDate:{state['target_date']}"
    r = quant_agent.invoke({"messages":[{"role":"user","content":prompt}]})
    return {"quant_analysis": r["messages"][-1].content}

def node_risk(state: PortfolioState):
    prompt = f"Calculate SL and TP for these tickers:{state['tickers']}"
    r = risk_agent.invoke({"messages":[{"role":"user","content":prompt}]})
    return {"risk_analysis": r["messages"][-1].content}

def node_trader(state: PortfolioState):
    prompt = f"""You are the Head Trader.Use all analyses below to construct a portfolio.
Fundamental:{state['fundamental_analysis']}
Technical:{state['technical_analysis']}
Sentiment:{state['sentiment_analysis']}
Quant:{state['quant_analysis']}
Risk:{state['risk_analysis']}
Return STRICT JSON.
Format:{{
"rationale":"short explanation",
"trades":{{
"AAPL":{{"action":"BUY","weight":0.2}},
"MSFT":{{"action":"SELL","weight":-0.1}}
}}
}}"""
    r = llm.invoke([{"role":"user","content":prompt}])
    try:
        json_str = re.search(r"\{.*\}", r.content, re.DOTALL).group()
        parsed = json.loads(json_str)
        return {
            "final_decision": parsed.get("rationale",""),
            "trade_book": parsed.get("trades",{})
        }
    except:
        fallback = {t: {"action":"SKIP","weight":0} for t in state["tickers"]}
        return {
            "final_decision":"JSON parsing error",
            "trade_book":fallback
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

graph.add_edge(START,"fundamentals")
graph.add_edge(START,"technicals")
graph.add_edge(START,"sentiment")
graph.add_edge(START,"quant")

graph.add_edge(["fundamentals","technicals","sentiment","quant"], "risk")
graph.add_edge("risk","trader")
graph.add_edge("trader",END)

app = graph.compile()