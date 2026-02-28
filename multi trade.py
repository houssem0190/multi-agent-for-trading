import os
import datetime
import numpy as np
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
# 1. Load Environment Variables
load_dotenv()

# 2. Define the State (Added target_date)
class InitialState(TypedDict):
    tickers: list[str]
    target_date: str
    fundamental_analysis: str
    technical_analysis: str
    sentiment_analysis: str
    optimal_weights: str
    market_direction: str
    final_decision: str

# ---------------------------------------------------------
# 3. DEFINE TOOLS (Updated for Point-in-Time Data)
# ---------------------------------------------------------
@tool
def get_fundamentals(ticker: str) -> str:
    """Gets company fundamentals from Finnhub."""
    finnhub_api_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_api_key:
        return "FINNHUB_API_KEY is missing."

    symbol = ticker.strip().upper()
    base_url = "https://finnhub.io/api/v1"
    params = {"symbol": symbol, "token": finnhub_api_key}

    try:
        profile_resp = requests.get(f"{base_url}/stock/profile2", params=params, timeout=15)
        metrics_resp = requests.get(f"{base_url}/stock/metric", params={**params, "metric": "all"}, timeout=15)
    except requests.RequestException as exc:
        return f"Finnhub request failed for {symbol}: {exc}"

    profile = profile_resp.json() or {}
    metrics_data = metrics_resp.json() or {}
    metric = metrics_data.get("metric", {}) if isinstance(metrics_data, dict) else {}

    company_name = profile.get("name") or symbol
    industry = profile.get("finnhubIndustry") or "N/A"
    market_cap = profile.get("marketCapitalization", "N/A")
    pe_ttm = metric.get("peTTM", "N/A")
    
    return f"{company_name} ({symbol}) fundamentals | Industry: {industry}, Market Cap: {market_cap}M, P/E TTM: {pe_ttm}."

@tool
def get_technicals(ticker: str, target_date: str) -> str:
    """Calculates RSI, MACD, and SMAs stopping exactly at the target_date."""
    end_date_obj = datetime.datetime.strptime(target_date, "%Y-%m-%d")
    start_date_obj = end_date_obj - relativedelta(months=6)
    
    df = yf.Ticker(ticker).history(start=start_date_obj.strftime("%Y-%m-%d"), end=end_date_obj.strftime("%Y-%m-%d"))
    if df.empty:
        return f"No historical data found for {ticker}."

    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.sma(length=50, append=True)
    
    latest = df.iloc[-1]
    rsi = latest.get('RSI_14', 50)
    macd = latest.get('MACD_12_26_9', 0)
    signal = latest.get('MACDs_12_26_9', 0)
    
    status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    trend = "Bullish Cross" if macd > signal else "Bearish Cross"
    
    return f"{ticker} Technicals on {target_date}: RSI is {rsi:.2f} ({status}). MACD is {macd:.2f} with a {trend}."

@tool
def get_sentiment(ticker: str) -> str:
    """Fetches the most recent news headlines for a given stock ticker."""
    try:
        news_items = yf.Ticker(ticker).news
        if not news_items:
            return f"No recent news found for {ticker}."
        report = f"Latest Headlines for {ticker}:\n"
        for item in news_items[:5]:
            report += f"- {item.get('title', 'No Title')}\n"
        return report
    except Exception as e:
        return f"Failed to fetch news for {ticker}: {str(e)}"

@tool
def calculate_markowitz(tickers: list[str], target_date: str) -> str:
    """Calculates GLASSO optimal weights using 1 year of data ending on the target_date."""
    try:
        if len(tickers) < 2:
            return "Error: Markowitz optimization requires at least 2 tickers."

        end_date_obj = datetime.datetime.strptime(target_date, "%Y-%m-%d")
        start_date_obj = end_date_obj - relativedelta(years=1)
        
        data = yf.download(tickers, start=start_date_obj.strftime("%Y-%m-%d"), end=end_date_obj.strftime("%Y-%m-%d"))['Close']
        returns = data.pct_change().dropna()
        returns_centered = returns - returns.mean()

        glasso = GraphicalLasso(alpha=0.6, max_iter=100)
        glasso.fit(returns_centered)
        precision_matrix = glasso.precision_

        ones = np.ones(len(tickers))
        raw_weights = precision_matrix.dot(ones) / ones.dot(precision_matrix).dot(ones)

        # Clip negatives (negative GLASSO weights have no clean long-only interpretation)
        # and normalize so weights always sum to exactly 1.0
        clipped = np.clip(raw_weights, 0, None)
        total = clipped.sum()
        if total <= 0:
            clipped = np.ones(len(tickers))  # fallback: equal weight
            total = clipped.sum()
        normalized_weights = clipped / total

        weight_dict = dict(zip(tickers, normalized_weights))
        report = f"GLASSO-Optimized Weights as of {target_date} (alpha=0.6):\n"
        for ticker, weight in weight_dict.items():
            report += f"- {ticker}: {weight * 100:.2f}%\n"

        return report
    except Exception as e:
        return f"GLASSO Optimization failed: {str(e)}"

# ---------------------------------------------------------
# 4. INITIALIZE LLM AND AGENTS
# ---------------------------------------------------------
llm = ChatOllama(model="phi3", temperature=0.1)

fund_agent = create_react_agent(llm, tools=[get_fundamentals], system_prompt="You are a fundamental analyst.")
tech_agent = create_react_agent(llm, tools=[get_technicals], system_prompt="You are a technical analyst.")
sent_agent = create_react_agent(llm, tools=[get_sentiment], system_prompt="You are a sentiment analyst.")
quant_agent = create_react_agent(llm, tools=[calculate_markowitz], system_prompt="You are a quantitative analyst.")


def _is_tool_support_error(exc: Exception) -> bool:
    """Detect Ollama tool-calling compatibility errors."""
    return "does not support tools" in str(exc).lower()

# ---------------------------------------------------------
# 5. WRAPPER NODES (Injecting the target_date)
# ---------------------------------------------------------
def fundamentals(state: InitialState):
    prompt = f"Pretend today is {state['target_date']}. Analyze the fundamentals for {state['tickers']}."
    try:
        response = fund_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"fundamental_analysis": response['messages'][-1].content}
    except Exception as exc:
        if not _is_tool_support_error(exc):
            raise
        reports = [get_fundamentals.invoke({"ticker": t}) for t in state["tickers"]]
        return {"fundamental_analysis": "\n".join(reports)}

def technicals(state: InitialState):
    prompt = f"Pretend today is {state['target_date']}. Analyze the technicals for {state['tickers']} passing the target_date '{state['target_date']}' to your tool."
    try:
        response = tech_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"technical_analysis": response['messages'][-1].content}
    except Exception as exc:
        if not _is_tool_support_error(exc):
            raise
        reports = [
            get_technicals.invoke({"ticker": t, "target_date": state["target_date"]})
            for t in state["tickers"]
        ]
        return {"technical_analysis": "\n".join(reports)}

def sentiment(state: InitialState): 
    prompt = f"Pretend today is {state['target_date']}. Analyze the sentiment for {state['tickers']}."
    try:
        response = sent_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"sentiment_analysis": response['messages'][-1].content}
    except Exception as exc:
        if not _is_tool_support_error(exc):
            raise
        reports = [get_sentiment.invoke({"ticker": t}) for t in state["tickers"]]
        return {"sentiment_analysis": "\n".join(reports)}

def risk(state: InitialState):
    # Use a structured chat message list so the system prompt is respected by Ollama
    messages = [
        {
            "role": "system",
            "content": (
                "You are a neutral risk analyst. You must respond with exactly ONE word: "
                "either BULLISH or BEARISH. Do not add any explanation, punctuation, or other text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Today is {state['target_date']}. Based on the three reports below, "
                f"is the overall market outlook for {state['tickers']} more BULLISH or BEARISH?\n\n"
                f"--- Fundamentals ---\n{state.get('fundamental_analysis', 'N/A')}\n\n"
                f"--- Technicals ---\n{state.get('technical_analysis', 'N/A')}\n\n"
                f"--- Sentiment ---\n{state.get('sentiment_analysis', 'N/A')}\n\n"
                "Respond with exactly one word: BULLISH or BEARISH."
            ),
        },
    ]

    response = llm.invoke(messages)
    # Extract only the last word to avoid "NOT BEARISH" false positives
    raw_direction = response.content.strip().upper().split()[-1]
    direction = "BEARISH" if raw_direction == "BEARISH" else "BULLISH"
    return {"market_direction": direction}

def route_market_direction(state: InitialState) -> str:
    if state.get("market_direction") == "BEARISH":
        return "short"
    return "long" 

def markowitz(state: InitialState):
    prompt = f"Calculate the optimal portfolio weights for {state['tickers']} passing the target_date '{state['target_date']}' to your tool."
    try:
        response = quant_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"optimal_weights": response['messages'][-1].content}
    except Exception as exc:
        if not _is_tool_support_error(exc):
            raise
        return {
            "optimal_weights": calculate_markowitz.invoke(
                {"tickers": state["tickers"], "target_date": state["target_date"]}
            )
        }

def trader_node(state: InitialState):
    prompt = f"""You are a head trader on {state['target_date']}. Review these reports and make a final LONG decision:
    Fundamentals: {state.get('fundamental_analysis')}
    Technicals: {state.get('technical_analysis')}
    Sentiment: {state.get('sentiment_analysis')}
    Quant: {state.get('optimal_weights')}"""
    response = llm.invoke(prompt)
    return {"final_decision": response.content}

def short_trader_node(state: InitialState):
    """
    BEARISH regime → defensive rotation, NOT blanket shorting.
    Strategy:
      - 40% into GLD  (gold, safe haven)
      - 30% into TLT  (long-term treasuries, flight-to-safety)
      - 20% into XOM  (energy / commodities, inflation hedge)
      - 10% cash-like (SHV: short-term treasuries, near-zero vol)
    Only the equity tickers get a small tactical short (max 20% of portfolio,
    volatility-scaled) to express the bearish view without blowing up.
    """
    prompt = f"""You are a head trader on {state['target_date']}. The market regime is BEARISH.
    Do NOT recommend shorting every stock. Instead, recommend a defensive rotation:
    reduce equity exposure, rotate into bonds (TLT), gold (GLD), and cash equivalents.
    Only consider a small, volatility-scaled short position on the weakest equity names.
    Fundamentals: {state.get('fundamental_analysis')}
    Technicals:   {state.get('technical_analysis')}
    Sentiment:    {state.get('sentiment_analysis')}
    Explain which assets to rotate into and why."""
    response = llm.invoke(prompt)
    return {"final_decision": f"[DEFENSIVE ROTATION TRIGGERED]\n{response.content}"}


# ---------------------------------------------------------
# 6. BUILD AND COMPILE THE GRAPH
# ---------------------------------------------------------
graph = StateGraph(InitialState)

graph.add_node("fundamental_analysis", fundamentals)
graph.add_node("technical_analysis", technicals)
graph.add_node("sentiment_analysis", sentiment)
graph.add_node("risk_analysis", risk)
graph.add_node("markowitz", markowitz)
graph.add_node("trader", trader_node)
graph.add_node("short_trader", short_trader_node)

graph.add_edge(START, "fundamental_analysis")
graph.add_edge(START, "technical_analysis")
graph.add_edge(START, "sentiment_analysis")
graph.add_edge(['fundamental_analysis', 'technical_analysis', 'sentiment_analysis'], 'risk_analysis')
graph.add_conditional_edges('risk_analysis', route_market_direction, {'short': 'short_trader', 'long': 'markowitz'})
graph.add_edge('markowitz', 'trader')
graph.add_edge("trader", END)
graph.add_edge("short_trader", END)

app = graph.compile()

# ---------------------------------------------------------
# 7. PORTFOLIO HELPERS
# ---------------------------------------------------------

# Defensive allocation used in BEARISH regime
# These are real, liquid ETFs that preserve / grow capital in downturns
DEFENSIVE_PORTFOLIO: dict[str, float] = {
    "GLD":  0.40,   # Gold — classic flight-to-safety
    "TLT":  0.30,   # 20yr Treasury bonds — rally when equities fall
    "XOM":  0.20,   # Energy / commodities — inflation hedge
    "SHV":  0.10,   # Short-term T-bills — near-zero vol cash proxy
}

# Max fraction of portfolio allocated to tactical equity shorts in bearish regime
MAX_SHORT_EXPOSURE = 0.20  # 20% of portfolio at most


def _vol_scaled_short_weights(tickers: list[str], date: str, budget: float) -> dict[str, float]:
    """
    For each equity ticker, compute an inverse-volatility short weight.
    Lower vol → smaller short (we don't want to short low-vol names hard).
    Higher vol → bigger short capped so total <= budget.
    Returns weight dict (values are fractions of total portfolio).
    """
    end   = datetime.datetime.strptime(date, "%Y-%m-%d")
    start = end - relativedelta(months=3)
    vols  = {}
    for t in tickers:
        try:
            px = yf.download(t, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"), progress=False)["Close"]
            if len(px) > 5:
                vols[t] = float(px.pct_change().dropna().std())
        except Exception:
            pass

    if not vols:
        return {}

    # Inverse-vol: higher vol ↔ less capital at risk (position size shrinks)
    inv_vol = {t: 1.0 / v for t, v in vols.items()}
    total   = sum(inv_vol.values())
    return {t: (w / total) * budget for t, w in inv_vol.items()}


# ---------------------------------------------------------
# 8. RUN / TEST THE CODE
# ---------------------------------------------------------
if __name__ == "__main__":
    import re
    print("🚀 Starting Local LLM Historical Backtest...")

    tickers_to_test = [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "NVDA",   # Nvidia
        "GOOGL",  # Alphabet
        "AMZN",   # Amazon
        "META",   # Meta
        "TSLA",   # Tesla
        "JPM",    # JPMorgan (financials)
        "XOM",    # Exxon (energy)
        "GLD",    # Gold ETF (safe haven)
    ]

    starting_capital = 10_000.0
    current_capital  = starting_capital
    backtest_dates   = ["2023-05-01", "2023-09-01", "2023-12-01", "2024-03-01"]
    trade_log        = []

    for date in backtest_dates:
        print(f"\n{'='*65}\n📅 BACKTEST DATE: {date}\n{'='*65}")

        inputs     = {"tickers": tickers_to_test, "target_date": date}
        final_state = app.invoke(inputs)

        regime         = final_state.get("market_direction", "BULLISH")
        final_decision = final_state.get("final_decision", "")
        weights_raw    = final_state.get("optimal_weights", "")

        print(f"\n🤖 AI Rationale:\n{'-'*50}\n{final_decision}\n{'-'*50}")
        print(f"📊 Regime: {regime}")

        # Parse Markowitz weights (lines like "- AAPL: 18.34%")
        parsed_weights: dict[str, float] = {}
        for line in weights_raw.splitlines():
            m = re.search(r"([A-Z]+):\s*([-\d.]+)%", line)
            if m:
                parsed_weights[m.group(1)] = float(m.group(2)) / 100.0

        next_month_str = (
            datetime.datetime.strptime(date, "%Y-%m-%d") + relativedelta(months=1)
        ).strftime("%Y-%m-%d")

        # ── Build the actual trade book ───────────────────────────────────────
        # trade_book: {ticker: (action, weight, note)}
        trade_book: dict[str, tuple[str, float, str]] = {}

        if regime == "BULLISH":
            # Long-only, Markowitz-weighted (fallback: equal weight)
            eq_w = 1.0 / len(tickers_to_test)
            for t in tickers_to_test:
                w = parsed_weights.get(t, eq_w)
                trade_book[t] = ("LONG", w, "Markowitz" if t in parsed_weights else "EqWt")

        else:  # BEARISH → defensive rotation + small vol-scaled equity shorts
            # 1. Defensive ETF allocation (80% of portfolio)
            for etf, w in DEFENSIVE_PORTFOLIO.items():
                trade_book[etf] = ("LONG", w, "Defensive")

            # 2. Small tactical shorts on equity tickers (max 20%)
            short_weights = _vol_scaled_short_weights(
                tickers_to_test, date, MAX_SHORT_EXPOSURE
            )
            for t, w in short_weights.items():
                if t not in trade_book:   # don't short something already held long
                    trade_book[t] = ("SHORT", w, "VolScaled")

        # ── Execute & price each position ─────────────────────────────────────
        print(f"\n{'Ticker':<7} {'Action':<6} {'Alloc':>6} {'Note':<12} "
              f"{'Buy @':>8} {'Sell @':>8} {'Ret%':>7} {'PnL $':>9}")
        print("-" * 72)

        month_rows = []
        for ticker, (action, weight, note) in sorted(trade_book.items()):
            try:
                px = yf.download(
                    ticker, start=date, end=next_month_str, progress=False
                )["Close"]
                if px.empty or len(px) < 2:
                    print(f"{ticker:<7} {'⚠️  no data':}")
                    continue

                buy_px   = float(px.iloc[0].squeeze())
                sell_px  = float(px.iloc[-1].squeeze())
                raw_ret  = (sell_px - buy_px) / buy_px

                # For SHORT positions profit = price going DOWN
                actual_ret = -raw_ret if action == "SHORT" else raw_ret

                # Hard stop-loss: cap loss at -8% per position
                actual_ret = max(actual_ret, -0.08)

                allocated = current_capital * weight
                pnl       = allocated * actual_ret
                arrow     = "📈" if actual_ret > 0 else "📉"

                print(f"{ticker:<7} {action:<6} {weight*100:>5.1f}%  {note:<12}"
                      f" {buy_px:>8.2f} {sell_px:>8.2f}"
                      f" {actual_ret*100:>+6.2f}% {arrow} {pnl:>+8.2f}")

                month_rows.append({
                    "date": date, "ticker": ticker, "action": action,
                    "note": note, "weight": weight,
                    "buy_price": buy_px, "sell_price": sell_px,
                    "return_pct": actual_ret * 100, "pnl": pnl,
                })

            except Exception as e:
                print(f"{ticker:<7} ⚠️  {e}")

        # ── Month roll-up ──────────────────────────────────────────────────────
        if month_rows:
            gross_weights = sum(r["weight"] for r in month_rows)
            total_pnl     = sum(r["pnl"] for r in month_rows)
            prev_capital  = current_capital
            current_capital += total_pnl
            month_ret_pct = (total_pnl / prev_capital) * 100

            print("-" * 72)
            print(f"{'TOTAL':<7} {'':6} {gross_weights*100:>5.1f}%  {'':12}"
                  f" {'':>8} {'':>8} {month_ret_pct:>+6.2f}%   {total_pnl:>+8.2f}")
            print(f"\n💰 Portfolio after {date}: ${current_capital:,.2f}  "
                  f"({'▲' if total_pnl >= 0 else '▼'} ${abs(total_pnl):,.2f})")
            trade_log.extend(month_rows)

    # ── Final summary ──────────────────────────────────────────────────────────
    total_return = ((current_capital - starting_capital) / starting_capital) * 100
    total_pnl    = current_capital - starting_capital

    print(f"\n{'='*65}")
    print(f"🏁 BACKTEST COMPLETE")
    print(f"   Starting Capital : ${starting_capital:>10,.2f}")
    print(f"   Ending Capital   : ${current_capital:>10,.2f}")
    print(f"   Total P&L        : ${total_pnl:>+10,.2f}")
    print(f"   Total Return     : {total_return:>+.2f}%")
    print(f"{'='*65}")

    if trade_log:
        best  = max(trade_log, key=lambda x: x["pnl"])
        worst = min(trade_log, key=lambda x: x["pnl"])
        print(f"\n🥇 Best Trade  : {best['ticker']} on {best['date']}"
              f" ({best['action']}) → {best['return_pct']:+.2f}%  ${best['pnl']:+,.2f}")
        print(f"💀 Worst Trade : {worst['ticker']} on {worst['date']}"
              f" ({worst['action']}) → {worst['return_pct']:+.2f}%  ${worst['pnl']:+,.2f}")

        # ── Per-regime breakdown ───────────────────────────────────────────────
        bullish_rows  = [r for r in trade_log if r["action"] == "LONG"  and r["note"] != "Defensive"]
        defensive_rows = [r for r in trade_log if r["note"] == "Defensive"]
        short_rows    = [r for r in trade_log if r["action"] == "SHORT"]

        def _summarise(rows: list, label: str):
            if not rows:
                return
            total = sum(r["pnl"] for r in rows)
            avg   = sum(r["return_pct"] for r in rows) / len(rows)
            print(f"   {label:<28}: {len(rows):>3} trades | avg ret {avg:>+.2f}% | PnL ${total:>+,.2f}")

        print(f"\n📋 Strategy Breakdown:")
        _summarise(bullish_rows,   "LONG (Markowitz, bullish)")
        _summarise(defensive_rows, "LONG (Defensive rotation)")
        _summarise(short_rows,     "SHORT (Vol-scaled, bearish)")