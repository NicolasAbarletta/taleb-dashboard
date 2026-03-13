# Taleb Trade Advisor

Live market opportunity scanner + trade recommendation engine built on Nassim Taleb's investment philosophy. Acts as a personal senior options trader on call.

## Setup (8 steps)

### 1. Requirements
- Python 3.10+

### 2. Create virtual environment
```bash
cd taleb-dashboard
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get API keys
| Key | Where to get it | Required? |
|-----|----------------|-----------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | For AI trade theses |
| `FRED_API_KEY` | [fred.stlouisfed.org/docs/api](https://fred.stlouisfed.org/docs/api/api_key.html) | For macro indicators |
| `NEWS_API_KEY` | [newsapi.org](https://newsapi.org/register) | For geopolitical signals |

### 5. Configure .env
```
ANTHROPIC_API_KEY=sk-ant-...
FRED_API_KEY=your_key
NEWS_API_KEY=your_key
```

### 6. Run
```bash
python run.py
```

### 7. Open dashboard
Navigate to `http://localhost:8501`

### 8. Stop
Press `Ctrl+C`

## Architecture

```
run.py            -- Orchestrator (bootstrap + agent loop + Streamlit)
agent.py          -- Data pulling (yfinance + FRED + NewsAPI)
scorer.py         -- Taleb scoring (4 filters, 0-100)
trade_builder.py  -- Trade recommendations (options, P&L, triggers)
dashboard.py      -- Streamlit UI
database.py       -- SQLite interface
```

## What it does

Every 30 minutes:
1. Scans 33 assets (ETFs, commodities, volatility products, defense, international)
2. Pulls 6 FRED macro indicators and 9 geopolitical news keywords
3. Scores each asset 0-100 across Convexity, Antifragility, Fragility Avoidance, Tail Risk
4. For assets scoring 50+, builds complete trade recommendations with options data, P&L scenarios, and payoff charts

## Not financial advice
For research and educational purposes only.
