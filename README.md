# Derivatives Terminal — FX & Equity Options Pricer

Lightweight Flask-based derivatives terminal for pricing and analysing FX and equity options.  
Provides live/historical market data, option-chain lookup, pricing engines, Greeks, scenario analysis and simple dashboards for quick risk checks.

What the terminal does
- Live & historical market data: fetches OHLC and latest close via yfinance with fallbacks (Stooq / sanitisation).
- Option chains: fetches chains via yfinance and falls back to scraping Yahoo Finance when needed.
- Pricing engines:
  - Black–Scholes (equities)
  - Garman–Kohlhagen (FX)
  - Binomial tree (European / American)
- Greeks: delta, gamma, vega, theta, rho (consistent outputs for degenerate inputs).
- Scenario analysis: run spot / vol / rate shocks and view P&L vs base.
- Model comparison: compare prices/Greeks across available models.
- Small in-memory caching for prices, chains and news to reduce external calls.
- Web UI: single-page Flask app with tabs for Chart, FX, Equity, Option Chains, Scenario and Comparison.

Features
- Real-time spot retrieval and historical OHLC (yfinance + robust fallbacks)
- Option chain fetch + parsing for ETFs / FX-related tickers
- Pricing: BS, GK, and binomial trees (configurable steps)
- Greeks and numerical Greeks (finite-difference) for model validation
- Scenario / stress testing (pre-built shock scenarios)
- News integration (NewsAPI) for headlines / sentiment (optional API key)
- Lightweight caching and error handling for unstable data sources

Requirements
- Python 3.8+
- Packages:
  - Flask
  - yfinance
  - pandas
  - numpy
  - requests
  - beautifulsoup4
  - (optional) NewsAPI key for news features

Installation (macOS / Linux)
1. Clone repository:
   git clone <repository-url>
   cd fx_options_pricer
2. Create and activate venv:
   python3 -m venv .venv
   source .venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
   (or) pip install flask yfinance pandas numpy requests beautifulsoup4

Configuration
- Set optional environment variables:
  - NEWS_API_KEY — API key for NewsAPI (optional; a placeholder is used if unset)
  - PORT — server port (default 5000)
- Tweak caching TTLs and defaults in app.py:
  - _PRICE_TTL, _CHAIN_TTL, _NEWS_TTL
- Currency ETF mapping for FX option-chain lookups is in CURRENCY_ETFS.

Usage
1. Start the app:
   python app.py
2. Open browser at:
   http://localhost:5000
3. Use the UI:
   - Chart tab: fetch OHLC and view recent history.
   - FX / Equity: enter symbol or enable "Use live price" to auto-fetch S, enter strikes / rates / vol and run pricing.
   - Chains: fetch option chains by symbol or currency.
   - Scenario: run preconfigured spot/vol/rate shocks and inspect P&L.
   - Comparison: compare models and inspect Greeks.

Project structure
- app.py — main application, pricing logic, data fetchers and routes
- templates/
  - index.html — UI (forms, charts)
- static/
  - css/, js/, assets for the frontend
- requirements.txt — Python dependencies
- README.md — this document

Developer notes / recommendations
- Use _truthy() normalization for form boolean flags to avoid truthiness issues.
- For production, remove debug mode and run under gunicorn or similar.
- Replace placeholder NEWS_API_KEY with an env variable before deploying.
- Add unit tests for pricing functions (edge cases: T=0, sigma=0, deep ITM/OTM).
- Increase binomial steps for accuracy; be mindful of UI timeout.

Disclaimer
This tool is for educational and informational purposes only. It is not financial advice. Always do your own research and consult a qualified financial professional before making investment decisions.
