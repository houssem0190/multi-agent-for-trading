# Multi-Agent Trading System

This project uses a directed graph (LangGraph) to orchestrate a team of specialized AI agents:
1. **The Technical Analyst:** Uses `yfinance` and `pandas_ta` to calculate Point-in-Time (PiT) indicators like RSI, MACD, and SMAs for a basket of stocks.
2. **The Risk Manager:** Evaluates the technical data to determine the current market regime (BULLISH vs. BEARISH).
3. **The Quantitative Analyst:** If the regime is Bullish, this agent executes a **Graphical Lasso (GLASSO)** algorithm to calculate the Global Minimum Variance portfolio weights using a 1-year historical covariance matrix.
4. **The Head Trader:** Synthesizes the regime and the mathematical weights to output a final trading execution strategy.
## Features
- Fundamental analysis (Finnhub)
- Technical analysis (yfinance + pandas-ta)
- Sentiment analysis (news headlines)
- Portfolio optimization (Graphical Lasso / Markowitz-style)
- Final trader decision node

## Tech Stack
- Python
- LangGraph
- LangChain
- Ollama (phi3)
- Finnhub API
- yfinance
- scikit-learn

## Setup

### 1) Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment variables
Create `.env` in project root:
```env
FINNHUB_API_KEY=your_finnhub_key
```

## Run
```bash
python "multi trade.py"
```

## Notes
- `.env` is ignored by git and should never be committed.
- This project is educational and not financial advice.

