# 🧠 Neura v4 — LSTM AI Retest Simulator

An AI-driven **Streamlit trading simulator** for **EMA9/EMA15 retest strategy**, powered by an **LSTM neural network** trained on live **Finnhub** data.  
This version is **fully local / educational**, designed for **testing strategies, model training, and performance analysis** — **no real trading** is executed.

---

## ⚙️ Features

✅ Fetches historical market data from **Finnhub** (XAUUSD default)  
✅ Supports **multiple timeframes** (`1m`, `5m`, `15m`, `1h`, `1D`)  
✅ Computes **EMA9/EMA15** and **ATR-based sideways detection**  
✅ Trains an **LSTM model** to classify TP vs SL outcomes  
✅ Automatically saves model weights in `/data/model_weights.h5`  
✅ Simulates trades with custom RR (`1:2` and `1:3`)  
✅ Logs all results to `/data/xau_trades_1_2.csv` and `/data/xau_trades_1_3.csv`  
✅ Displays **live chart** + **focused trade chart**  
✅ Auto retrain and manual retrain options  
✅ Works with any Finnhub-supported asset (`OANDA:XAUUSD`, `OANDA:BTCUSD`, etc.)

---

## 🏗️ Project Structure


---

## 🧩 Setup

### 1. Clone the repo
```bash
git clone https://github.com/<yourname>/neura_v4.git
cd neura_v4

2. Install dependencies
pip install -r requirements.txt

3. Set your Finnhub API key

Either:

Export it:

export FINNHUB_KEY="your_finnhub_api_key"


or create a file:

data/finnhub_key.txt

4. Run locally
streamlit run app.py

5. Deploy on Render

Add:

Build Command: pip install -r requirements.txt

Start Command: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0

Environment Variable:

FINNHUB_KEY = your API key

📊 Usage

Fetch candles → downloads Finnhub data.

Train LSTM → builds/updates model with retest-based signals.

Simulate all setups → generates trades with SL/TP.

Trade history → shows log + focused chart.

Manual LSTM test → check model probability on any candle index.

⚠️ Disclaimer

Neura v4 is a learning and testing platform, not a trading bot.
It does not execute live orders or connect to real brokers.
All simulations are hypothetical and for educational use only.

Author: Harshavardhan G R
Version: 4.0 (LSTM Edition)


---

Would you like me to include a **sample `btcusd_data.csv`** (for offline demo without Finnhub key)?  
It’ll let you test the LSTM logic instantly before connecting to Finnhub.