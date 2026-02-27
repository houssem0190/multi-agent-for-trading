import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from matplotlib import ticker
from sklearn.covariance import GraphicalLasso
from langchain_core.tools import tool
import numpy as np
import yfinance as yf
import pandas_ta as ta
import requests
load_dotenv()

class InitialState(TypedDict):
    tickers: list[str]
    fundamental_analysis: str
    technical_analysis: str
    sentiment_analysis: str
    optimal_weights: str
    market_direction: str
    final_decision: str

# 2. Mock Tools (You will fill these with real API calls later)
@tool
def get_fundamentals(ticker: str) -> str:
    """Gets company fundamentals from Finnhub."""
    finnhub_api_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_api_key:
        return (
            "FINNHUB_API_KEY is missing. Add it to .env to enable live fundamental analysis."
        )

    symbol = ticker.strip().upper()
    base_url = "https://finnhub.io/api/v1"
    params = {"symbol": symbol, "token": finnhub_api_key}

    try:
        profile_resp = requests.get(
            f"{base_url}/stock/profile2",
            params=params,
            timeout=15,
        )
        metrics_resp = requests.get(
            f"{base_url}/stock/metric",
            params={**params, "metric": "all"},
            timeout=15,
        )
    except requests.RequestException as exc:
        return f"Finnhub request failed for {symbol}: {exc}"

    if profile_resp.status_code != 200 or metrics_resp.status_code != 200:
        return (
            f"Finnhub returned an error for {symbol}. "
            f"profile={profile_resp.status_code}, metric={metrics_resp.status_code}"
        )

    profile = profile_resp.json() or {}
    metrics_data = metrics_resp.json() or {}
    metric = metrics_data.get("metric", {}) if isinstance(metrics_data, dict) else {}

    company_name = profile.get("name") or symbol
    industry = profile.get("finnhubIndustry") or "N/A"
    market_cap = profile.get("marketCapitalization")
    pe_ttm = metric.get("peTTM")
    pb = metric.get("pbAnnual")
    eps_ttm = metric.get("epsTTM")
    roe_ttm = metric.get("roeTTM")
    net_margin = metric.get("netMarginAnnual")
    rev_growth_5y = metric.get("revenueGrowth5Y")

    return (
        f"{company_name} ({symbol}) fundamentals | "
        f"Industry: {industry}, "
        f"Market Cap: {market_cap if market_cap is not None else 'N/A'}M, "
        f"P/E TTM: {pe_ttm if pe_ttm is not None else 'N/A'}, "
        f"P/B: {pb if pb is not None else 'N/A'}, "
        f"EPS TTM: {eps_ttm if eps_ttm is not None else 'N/A'}, "
        f"ROE TTM: {roe_ttm if roe_ttm is not None else 'N/A'}, "
        f"Net Margin: {net_margin if net_margin is not None else 'N/A'}, "
        f"5Y Revenue Growth: {rev_growth_5y if rev_growth_5y is not None else 'N/A'}."
    )


@tool
def get_technicals(ticker: str) -> str:
    """Calculates RSI, MACD, and 50/200-day Moving Averages using yfinance."""
    df = yf.Ticker(ticker).history(period="6mo")
    if df.empty:
        return f"No data found for {ticker}."

    # Calculate indicators using pandas-ta extension
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.sma(length=50, append=True)
    
    latest = df.iloc[-1]
    rsi = latest['RSI_14']
    macd = latest['MACD_12_26_9']
    signal = latest['MACDs_12_26_9']
    
    status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    trend = "Bullish Cross" if macd > signal else "Bearish Cross"
    
    return (f"{ticker} Technicals: RSI is {rsi:.2f} ({status}). "
            f"MACD is {macd:.2f} with a {trend}. "
            f"Current Price: {latest['Close']:.2f}.")


@tool
def get_sentiment(ticker: str) -> str:
    """
    Fetches the most recent news headlines and publishers for a given stock ticker.
    Use this to gather raw data for sentiment analysis.
    """
    try:
        # yfinance has a built-in news property!
        news_items = yf.Ticker(ticker).news
        
        if not news_items:
            return f"No recent news found for {ticker}."
        
        report = f"Latest Headlines for {ticker}:\n"
        
        # Grab the top 5 most recent articles
        for item in news_items[:5]:
            title = item.get('title', 'No Title')
            publisher = item.get('publisher', 'Unknown Publisher')
            report += f"- [{publisher}] {title}\n"
            
        return report

    except Exception as e:
        return f"Failed to fetch news for {ticker}: {str(e)}"



@tool
def calculate_markowitz(tickers: list[str]) -> str:
    """
    Downloads historical data, calculates a sparse precision matrix using 
    Graphical Lasso (alpha=0.6), and returns optimal Global Minimum Variance weights.
    """
    try:
        if len(tickers) < 2:
            return "Error: Markowitz optimization requires at least 2 tickers."

        data = yf.download(tickers, period="1y")['Close']
        returns = data.pct_change().dropna()

        
        returns_centered = returns - returns.mean()

        glasso = GraphicalLasso(alpha=0.6, max_iter=100)
        glasso.fit(returns_centered)
        precision_matrix = glasso.precision_

        ones = np.ones(len(tickers))
        raw_weights = precision_matrix.dot(ones) / ones.dot(precision_matrix).dot(ones)

        weight_dict = dict(zip(tickers, raw_weights))
        
        report = "GLASSO-Optimized Portfolio Weights (alpha=0.6):\n"
        for ticker, weight in weight_dict.items():
            report += f"- {ticker}: {weight * 100:.2f}%\n"
            
        return report

    except Exception as e:
        return f"GLASSO Optimization failed: {str(e)}"


 
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing. Put it in a local .env file or set an environment variable.")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=api_key)
fund_agent = create_react_agent(llm, tools=[get_fundamentals], state_modifier="You are a fundamental analyst. Summarize the fundamentals.")
tech_agent = create_react_agent(llm, tools=[get_technicals], state_modifier="You are a technical analyst. Summarize the chart indicators.")
sent_agent = create_react_agent(llm, tools=[get_sentiment], state_modifier="You are a sentiment analyst. Summarize the news.")
quant_agent = create_react_agent(llm, tools=[calculate_markowitz], state_modifier="You are a quantitative analyst. Output the weights.")
def markowitz(state: InitialState):
    # 1. Read the tickers from the state clipboard
    tickers_to_analyze = state["tickers"]
    
    # 2. Tell the agent what to do
    message = f"Please calculate the optimal weights for these tickers: {tickers_to_analyze}"
    
    # 3. Run the ReAct agent
    response = quant_agent.invoke({"messages": [("user", message)]})
    
    # 4. Extract the agent's final answer and return it to update the state
    final_output = response["messages"][-1].content
    return {"optimal_weights": final_output}
def fundamentals(state: InitialState):
    tickers = state['tickers']
    message = f"please analyze the fundamentals of these tickers: {tickers}"
    response = fund_agent.invoke({"messages":[("user",message)]})
    return{"fundamental analysis":response['messages'][-1].content}
def technicals(state: InitialState):
    tickers = state['tickers']
    message = f"please analyze the technicals of these tickers: {tickers}"
    response = tech_agent.invoke({"messages":[("user",message)]})
    return{"technical analysis":response['messages'][-1].content}
def sentiment(state: InitialState): 
    tickers = state['tickers']
    message = f"please analyze the sentiment of these tickers: {tickers}"
    response = sent_agent.invoke({"messages":[("user",message)]})
    return{"sentiment analysis":response['messages'][-1].content}
def risk(state:InitialState):
    message = f"""You are a risk analyst . Review the reports of {state['tickers']}:
    Fundamentals: {state.get('fundamental_analysis')}
    Technicals: {state.get('technical_analysis')}
    Sentiment: {state.get('sentiment_analysis')}
    If the overall consensus is clearly negative/bearish, output EXACTLY the word "BEARISH".
    Otherwise, if it is neutral or positive, output EXACTLY the word "BULLISH"."""
    response = llm.invoke(message)
    direction = response.content.strip().upper()
    if direction != "BEARISH":
        direction = "BULLISH"
    return {"market_direction": direction}  
def route_market_direction(state: InitialState) -> str:
    """Checks the market direction and routes the graph."""
    if state.get("market_direction") == "BEARISH":
        return "short"
    else:
        return "long" 
def trader_node(state: InitialState):
    # This node reads all previous analysis from the state
    prompt = f"""You are a head trader. Review these reports and make a final decision only long is allowed :
    Fundamentals: {state.get('fundamental_analysis')}
    Technicals: {state.get('technical_analysis')}
    Sentiment: {state.get('sentiment_analysis')}
    Quant: {state.get('optimal_weights')}"""
    
    response = llm.invoke(prompt)
    return {"final_decision": response.content}

graph = StateGraph[InitialState]()
graph.add_node(START, "start")
graph.add_node("fundamental_analysis",fundamentals)
graph.add_node("technical_analysis",technicals)
graph.add_node("sentiment_analysis",sentiment)
graph.add_node("Markowitz", markowitz)
graph.add_node("trader", trader_node)
graph.add_node(END, "end")
graph.add_node('risk_analysis', risk)
graph.add_edge(START, "fundamental_analysis")
graph.add_edge(START, "technical_analysis")    
graph.add_edge(START, "sentiment_analysis")
graph.add_edge(['fundamental_analysis', 'technical_analysis', 'sentiment_analysis'], 'risk_analysis')
graph.add_conditional_edges('risk_analysis', route_market_direction, {'short':'short trader','long':'markowitz'})
graph.add_edge('markowitz', 'trader')
graph.add_edge("trader", END, lambda state: True)
app = graph.compile()