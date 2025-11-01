# neura_v4_lstm.py
"""
Neura v4.0 (LSTM) — Streamlit app
- Finnhub data source (uses FINNHUB_KEY environment variable or data/finnhub_key.txt)
- LSTM-based model to predict retest success (TP vs SL)
- Multi-timeframe support
- Sideways detection (ATR-based) and skip trades in sideways markets
- Pip/points rules per symbol (BTC special rules)
- Trade simulation logs: xau_trades_1_2.csv, xau_trades_1_3.csv, xau_orders_sim.csv
- Single main chart view + focused chart below when a trade is selected in table
- No email/login (anonymous)
"""
import os
import json
import time
import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

import plotly.graph_objects as go
import streamlit as st

# Machine learning / LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
import os
import finnhub
import pandas as pd

st.set_page_config(page_title="Neura v4 - LSTM AI Trading", layout="wide")

st.title("🧠 Neura v4 — LSTM Trading AI (Finnhub Live)")
st.write("Status check running...")

# --- Verify API Key ---
api_key = os.getenv("FINNHUB_KEY")
if not api_key:
    st.error("❌ Environment variable FINNHUB_KEY not found! Please add it in Render > Environment.")
else:
    st.success("✅ FINNHUB_KEY loaded.")
    try:
        finnhub_client = finnhub.Client(api_key=api_key)
        quote = finnhub_client.quote("XAUUSD")
        if not quote or quote.get('c') == 0:
            st.warning("⚠️ Finnhub API returned no data (maybe rate limit or symbol issue).")
        else:
            st.success("✅ Finnhub connected successfully!")
            st.json(quote)
    except Exception as e:
        st.error(f"Finnhub connection failed: {e}")

# Prevent blank screen — placeholder message
st.markdown("---")
st.info("✅ App backend is working. If you see this message, Streamlit UI is rendering properly.")

# --------------------- Helpers & Config ---------------------
APP_TITLE = "Neura v4.0 — LSTM Retest Simulator (Finnhub)"
DATA_DIR = "data"
MODEL_DIR = os.path.join(DATA_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "neura_lstm.h5")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.npy")
FINNHUB_KEY_FILE = os.path.join(DATA_DIR, "finnhub_key.txt")

# logs / csv files
TRADES_CSV_1_2 = os.path.join(DATA_DIR, "xau_trades_1_2.csv")
TRADES_CSV_1_3 = os.path.join(DATA_DIR, "xau_trades_1_3.csv")
ORDERS_CSV = os.path.join(DATA_DIR, "xau_orders_sim.csv")
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_crosses_v4.json")

# finnub default symbol mapping (use OANDA pair for forex/commodities)
DEFAULT_SYMBOL = "OANDA:XAUUSD"  # XAUUSD default
AVAILABLE_TIMEFRAMES = ["1", "5", "15", "60", "D"]  # minutes (string) and D for daily
LOOKBACK = 60  # LSTM lookback (candles)
LSTM_EPOCHS = 30
LSTM_BATCH = 64

# trade sim defaults
DEFAULT_SLR_PTS = 5.0
DEFAULT_MAX_HOLD = 48
BTC_SL_POINTS = 500.0   # points for BTC (as user requested)
BTC_TP_POINTS = 1000.0
# We'll treat 'points' as raw price units for simplicity (user can adapt to pip sizes if needed)

# ensure directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------- Utility functions ---------------------
def safe_rerun():
    """Robust rerun for Streamlit versions."""
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        pass
    try:
        if hasattr(st, "rerun"):
            st.rerun()
            return
    except Exception:
        pass
    return

def read_finnhub_key():
    # 1) environment variable
    env = os.getenv("FINNHUB_KEY")
    if env:
        return env.strip()
    # 2) file fallback
    if os.path.exists(FINNHUB_KEY_FILE):
        try:
            with open(FINNHUB_KEY_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return None
    return None

FINNHUB_KEY = read_finnhub_key()

def finnhub_candles(symbol: str, resolution: str, _from: int, to: int, token: str):
    base = "https://finnhub.io/api/v1/forex/candle"
    params = {"symbol": symbol, "resolution": resolution, "from": int(_from), "to": int(to), "token": token}
    try:
        r = requests.get(base, params=params, timeout=20)
    except Exception as e:
        return None
    if r.status_code != 200:
        return None
    return r.json()

def fetch_candles_finnhub(symbol=DEFAULT_SYMBOL, resolution="15", days=90):
    """Fetch historical candles from Finnhub."""
    if not FINNHUB_KEY:
        return pd.DataFrame()
    now_ts = int(time.time())
    # estimate from_ts using days (rough)
    from_ts = now_ts - int(days*24*3600)
    data = finnhub_candles(symbol, resolution, from_ts, now_ts, FINNHUB_KEY)
    if not data or data.get("s") != "ok":
        return pd.DataFrame()
    df = pd.DataFrame({
        "time": data["t"],
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
        "volume": data.get("v", [0]*len(data["t"]))
    })
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df = df[['date','open','high','low','close','volume']]
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def fetch_latest_n_candles(symbol=DEFAULT_SYMBOL, resolution="15", n=500):
    """Fetch latest n candles (approx) from Finnhub."""
    if not FINNHUB_KEY:
        return pd.DataFrame()
    try:
        res_min = 1 if resolution == "1" else (5 if resolution=="5" else (15 if resolution=="15" else (60 if resolution=="60" else 24*60)))
        secs_needed = n * res_min * 60
    except Exception:
        secs_needed = n * 60 * 60
    now_ts = int(time.time())
    from_ts = now_ts - secs_needed - 3600
    data = finnhub_candles(symbol, resolution, from_ts, now_ts, FINNHUB_KEY)
    if not data or data.get("s") != "ok":
        return pd.DataFrame()
    df = pd.DataFrame({
        "time": data["t"],
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
        "volume": data.get("v", [0]*len(data["t"]))
    })
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df = df[['date','open','high','low','close','volume']]
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    if len(df) > n:
        return df.tail(n).reset_index(drop=True)
    return df

# ------------------- Indicators -------------------
def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, window=14):
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prev_close']).abs()
    df['tr3'] = (df['low'] - df['prev_close']).abs()
    df['tr'] = df[['tr1','tr2','tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=window, min_periods=1).mean()
    return df['atr']

def compute_indicators(df: pd.DataFrame):
    df = df.copy()
    df['EMA9'] = ema(df['close'], 9)
    df['EMA15'] = ema(df['close'], 15)
    df['ema_diff'] = df['EMA9'] - df['EMA15']
    df['atr_14'] = atr(df, 14)
    df['range'] = df['high'] - df['low']
    df['ret1'] = df['close'].pct_change().fillna(0)
    df['vol_5'] = df['close'].rolling(5).std().fillna(0)
    return df

def detect_crosses(df: pd.DataFrame):
    if df.empty:
        return df
    df = compute_indicators(df)
    df['cross'] = None
    for i in range(1, len(df)):
        prev_s = df.loc[i-1,'EMA9']; prev_l = df.loc[i-1,'EMA15']
        curr_s = df.loc[i,'EMA9']; curr_l = df.loc[i,'EMA15']
        if pd.isna(prev_s) or pd.isna(prev_l) or pd.isna(curr_s) or pd.isna(curr_l):
            continue
        if prev_s <= prev_l and curr_s > curr_l:
            df.at[i,'cross'] = 'bull'
        elif prev_s >= prev_l and curr_s < curr_l:
            df.at[i,'cross'] = 'bear'
    return df

def find_first_retest(df: pd.DataFrame, cross_idx: int, lookahead=12):
    if pd.isna(df.loc[cross_idx,'cross']):
        return None
    direction = df.loc[cross_idx,'cross']
    for j in range(cross_idx+1, min(cross_idx+1+lookahead, len(df))):
        for ema_col in ('EMA9','EMA15'):
            ema_val = float(df.loc[j, ema_col])
            low, high, close = float(df.loc[j,'low']), float(df.loc[j,'high']), float(df.loc[j,'close'])
            if direction == 'bull':
                if low <= ema_val and close >= ema_val:
                    return j, close, ema_col
            else:
                if high >= ema_val and close <= ema_val:
                    return j, close, ema_col
    return None

# ------------------- Model utilities (LSTM) -------------------
def create_sequences(df: pd.DataFrame, features: list, lookback=LOOKBACK):
    """
    Build sequences for LSTM.
    df: must contain computed indicators
    features: list of column names to use
    returns X (n, lookback, features), indices (the index of the last candle in each sequence)
    """
    X = []
    idxs = []
    if len(df) < lookback + 1:
        return np.array(X), idxs
    arr = df[features].values
    for i in range(lookback, len(df)):
        seq = arr[i-lookback:i]
        X.append(seq)
        idxs.append(i)
    return np.array(X), idxs

def build_label_for_index(df: pd.DataFrame, index: int, direction: str, rr='1:2', sl_points=DEFAULT_SLR_PTS, max_hold=DEFAULT_MAX_HOLD):
    """
    Determine whether a retest-based entry at 'index' would reach TP or SL.
    Returns label 1 (TP) or 0 (SL) and exit details.
    """
    entry_price = float(df.loc[index,'close'])
    if rr == '1:2':
        tp_distance = sl_points * 2.0
    elif rr == '1:3':
        tp_distance = sl_points * 3.0
    else:
        # if rr numeric factor provided like '2.5' treat as multiplier
        try:
            factor = float(rr)
            tp_distance = sl_points * factor
        except Exception:
            tp_distance = sl_points * 2.0

    if direction == 'bull':
        sl_price = entry_price - sl_points
        tp_price = entry_price + tp_distance
    else:
        sl_price = entry_price + sl_points
        tp_price = entry_price - tp_distance

    exit_reason = "Timeout"; exit_price = float(df.loc[min(index+max_hold, len(df)-1),'close']); exit_idx = min(index+max_hold, len(df)-1)
    for k in range(index+1, min(index+1+int(max_hold), len(df))):
        high = float(df.loc[k,'high']); low = float(df.loc[k,'low'])
        if direction == 'bull':
            if high >= tp_price:
                exit_reason = "TP"; exit_price = tp_price; exit_idx = k; break
            if low <= sl_price:
                exit_reason = "SL"; exit_price = sl_price; exit_idx = k; break
        else:
            if low <= tp_price:
                exit_reason = "TP"; exit_price = tp_price; exit_idx = k; break
            if high >= sl_price:
                exit_reason = "SL"; exit_price = sl_price; exit_idx = k; break
    pnl = (exit_price - entry_price) if direction == 'bull' else (entry_price - exit_price)
    label = 1 if exit_reason == "TP" else 0
    return label, exit_reason, exit_price, exit_idx, pnl

def prepare_dataset_for_lstm(df: pd.DataFrame, lookback=LOOKBACK, rr='1:2', sl_points=DEFAULT_SLR_PTS, sideways_atr_thresh=None):
    """
    Build training dataset using retest signals.
    Returns X (sequences), y (labels), meta dataframe of entries.
    """
    dfc = df.copy().reset_index(drop=True)
    dfc = compute_indicators(dfc)
    dfc = detect_crosses(dfc)
    features = ['close','EMA9','EMA15','ema_diff','atr_14','ret1','vol_5','range']
    X_rows = []
    y = []
    metas = []
    # look for cross indices and retest
    for i in dfc.index:
        if pd.isna(dfc.loc[i,'cross']):
            continue
        # sideways check before taking this setup if threshold provided
        if sideways_atr_thresh is not None:
            # use ATR at the cross index
            atr_val = float(dfc.loc[i,'atr_14']) if not pd.isna(dfc.loc[i,'atr_14']) else 0.0
            if atr_val < sideways_atr_thresh:
                # skip sideways
                continue
        ret = find_first_retest(dfc, i, lookahead=12)
        if ret is None:
            continue
        ret_idx, ret_price, used_ema = ret
        label, exit_reason, exit_price, exit_idx, pnl = build_label_for_index(dfc, ret_idx, dfc.loc[i,'cross'], rr=rr, sl_points=sl_points)
        # create sequence ending at ret_idx
        if ret_idx < lookback:
            continue
        seq = dfc.loc[ret_idx-lookback+1:ret_idx+1, features].values  # shape (lookback, features)
        if seq.shape[0] != lookback:
            continue
        X_rows.append(seq)
        y.append(label)
        metas.append({
            'cross_idx': int(i),
            'entry_idx': int(ret_idx),
            'entry_time': dfc.loc[ret_idx,'date'],
            'direction': dfc.loc[i,'cross'],
            'entry_price': float(ret_price),
            'used_ema': used_ema,
            'exit_reason': exit_reason,
            'exit_price': exit_price,
            'pnl': pnl
        })
    if not X_rows:
        return np.array([]), np.array([]), pd.DataFrame(metas)
    X = np.array(X_rows)
    y = np.array(y)
    meta_df = pd.DataFrame(metas)
    return X, y, meta_df

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ------------------- Persistence helpers -------------------
def ensure_csv_with_header(path, header):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=header)
        df.to_csv(path, index=False)

ensure_csv_with_header(TRADES_CSV_1_2, ["entry_time","entry_index","exit_time","exit_index","direction","entry_price","exit_price","exit_reason","pnl","used_ema","rr"])
ensure_csv_with_header(TRADES_CSV_1_3, ["entry_time","entry_index","exit_time","exit_index","direction","entry_price","exit_price","exit_reason","pnl","used_ema","rr"])
ensure_csv_with_header(ORDERS_CSV, ["time","symbol","direction","entry_price","sl","tp","lot","event","note"])

def append_trade_csv(trade: dict, rr_choice='1:2'):
    path = TRADES_CSV_1_2 if rr_choice == '1:2' else TRADES_CSV_1_3
    header = not os.path.exists(path)
    pd.DataFrame([trade]).to_csv(path, mode='a', index=False, header=header)

def append_order_log(order_info: dict):
    header = not os.path.exists(ORDERS_CSV)
    pd.DataFrame([order_info]).to_csv(ORDERS_CSV, mode='a', index=False, header=header)

def load_trades(rr_choice='1:2'):
    path = TRADES_CSV_1_2 if rr_choice == '1:2' else TRADES_CSV_1_3
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()

# ------------------- Sideways detection (ATR-based) -------------------
def is_sideways(df: pd.DataFrame, atr_window=14, atr_thresh=0.5):
    """
    Simple sideways detection: compare ATR to recent price scale.
    atr_thresh is ratio of ATR to mean close (e.g., 0.002 = 0.2%).
    Returns boolean.
    """
    dfc = df.copy()
    dfc['atr'] = atr(dfc, window=atr_window)
    mean_close = dfc['close'].tail(atr_window).mean() if len(dfc) >= atr_window else dfc['close'].mean()
    current_atr = dfc['atr'].iloc[-1] if not dfc['atr'].isna().all() else 0.0
    if mean_close == 0:
        return False
    ratio = current_atr / mean_close
    return ratio < atr_thresh

# ------------------- UI and App Layout -------------------
st.set_page_config(layout="wide", page_title=APP_TITLE)
st.title(APP_TITLE)
st.markdown("**No login required — LSTM-based simulated retest / backtest environment.**")

# Sidebar controls
st.sidebar.header("Data & Model Settings")
symbol_input = st.sidebar.text_input("Finnhub symbol (e.g. OANDA:XAUUSD, OANDA:BTCUSD)", value=DEFAULT_SYMBOL)
timeframe = st.sidebar.selectbox("Timeframe (minutes)", options=AVAILABLE_TIMEFRAMES, index=2)
days_back = st.sidebar.number_input("History days to fetch", min_value=1, max_value=365, value=90)
candles_show = st.sidebar.selectbox("Candles to show on chart", options=[100,200,400,800], index=1)
lookback = st.sidebar.number_input("LSTM lookback (candles)", min_value=10, max_value=240, value=LOOKBACK)
sl_points_default = st.sidebar.number_input("Default SL points (for most symbols)", value=DEFAULT_SLR_PTS)
max_hold = st.sidebar.number_input("Max hold candles (sim)", min_value=1, max_value=500, value=DEFAULT_MAX_HOLD)
sideways_atr_ratio = st.sidebar.slider("Sideways ATR ratio threshold (smaller=more sideways)", 0.0001, 0.02, 0.002, step=0.0001)
rr_options = st.sidebar.multiselect("Simulate RRs", options=['1:2','1:3'], default=['1:2','1:3'])
train_epochs = st.sidebar.number_input("LSTM epochs", min_value=1, max_value=200, value=LSTM_EPOCHS)
auto_retrain = st.sidebar.checkbox("Auto retrain after logging new trades", value=False)

st.sidebar.markdown("---")
st.sidebar.header("Actions")
fetch_btn = st.sidebar.button("Fetch candles")
refresh_btn = st.sidebar.button("Refresh UI")
train_btn = st.sidebar.button("Train LSTM")
simulate_btn = st.sidebar.button("Simulate all setups (from data)")
download_trades_btn = st.sidebar.button("Download trades (1:2 CSV)")
download_trades_1_3_btn = st.sidebar.button("Download trades (1:3 CSV)")
clear_trades_btn = st.sidebar.button("Clear all simulated trades (CSV)")

# display finnhub key status
if FINNHUB_KEY:
    st.sidebar.success("Finnhub key detected")
else:
    st.sidebar.error("Finnhub key missing. Set FINNHUB_KEY env var or create data/finnhub_key.txt")

# main area: fetch data if requested or show last cached if present
df = pd.DataFrame()
cache_file = os.path.join(DATA_DIR, f"cached_{symbol_input.replace(':','_')}_{timeframe}.csv")
if fetch_btn or (not os.path.exists(cache_file)):
    with st.spinner("Fetching candles from Finnhub..."):
        df = fetch_candles_finnhub(symbol=symbol_input, resolution=timeframe, days=int(days_back))
        if df.empty:
            st.error("No data fetched from Finnhub. Check symbol and API key.")
        else:
            df.to_csv(cache_file, index=False)
else:
    try:
        df = pd.read_csv(cache_file, parse_dates=['date'])
    except Exception:
        df = pd.DataFrame()

# allow manual refresh of UI (no data fetch)
if refresh_btn:
    safe_rerun()

if df.empty:
    st.warning("No candle data available. Use sidebar 'Fetch candles' or provide Finnhub API key.")
    st.stop()

# compute indicators & crosses
df = compute_indicators(df)
df = detect_crosses(df)

# show main chart
st.markdown("## Main chart")
display_df = df.tail(candles_show).copy().reset_index(drop=True)
fig = go.Figure()
fig.add_trace(go.Candlestick(x=display_df['date'], open=display_df['open'], high=display_df['high'], low=display_df['low'], close=display_df['close'], name='price'))
fig.add_trace(go.Scatter(x=display_df['date'], y=display_df['EMA9'], mode='lines', name='EMA9'))
fig.add_trace(go.Scatter(x=display_df['date'], y=display_df['EMA15'], mode='lines', name='EMA15'))
fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(fig, use_container_width=True)

# Build dataset for LSTM training
with st.expander("Dataset & labeling (prepare for LSTM)"):
    st.write("Preparing labeled dataset from retest setups (skip sideways setups based on ATR threshold).")
    X_seq, y_labels, meta_df = prepare_dataset_for_lstm(df, lookback=int(lookback), rr='1:2', sl_points=float(sl_points_default), sideways_atr_thresh=float(sideways_atr_ratio))
    if X_seq.size == 0:
        st.info("No labeled setups found with current settings. Try increasing history days, lowering sideways threshold, or changing timeframe.")
    else:
        st.success(f"Found {len(y_labels)} labeled setups for training.")
        st.dataframe(meta_df.sort_values('entry_time', ascending=False).head(20))

# Model training
model_loaded = None
if os.path.exists(MODEL_FILE):
    try:
        model_loaded = load_model(MODEL_FILE)
    except Exception:
        model_loaded = None

st.markdown("## Model (LSTM)")
col_m1, col_m2 = st.columns([2,1])
with col_m1:
    if model_loaded is not None:
        st.success("Saved LSTM model loaded.")
        if st.button("Show model summary"):
            try:
                st.text(str(model_loaded.summary()))
            except Exception as e:
                st.write("Could not display model summary:", e)
    else:
        st.info("No saved model found. Train model using 'Train LSTM' in sidebar.")

with col_m2:
    if train_btn:
        if X_seq.size == 0:
            st.warning("No training data available. Adjust settings or fetch more history.")
        else:
            with st.spinner("Training LSTM model..."):
                # flatten features for scaler: reshape to (n_samples, seq_len*features) for scaling across features
                nsamples, seq_len, nfeat = X_seq.shape
                X_flat = X_seq.reshape((nsamples, seq_len*nfeat))
                scaler = StandardScaler().fit(X_flat)
                X_scaled = scaler.transform(X_flat).reshape((nsamples, seq_len, nfeat))
                # save scaler
                np.save(SCALER_FILE, scaler.mean_), np.save(SCALER_FILE + ".scale", scaler.scale_)
                # split
                split_idx = int(nsamples*0.8)
                X_train = X_scaled[:split_idx]; X_test = X_scaled[split_idx:]
                y_train = y_labels[:split_idx]; y_test = y_labels[split_idx:]
                model = build_lstm_model((seq_len, nfeat))
                cb_early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                cb_chk = ModelCheckpoint(MODEL_FILE, monitor='val_loss', save_best_only=True, save_weights_only=False)
                history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=int(train_epochs), batch_size=int(LSTM_BATCH), callbacks=[cb_early, cb_chk], verbose=0)
                st.success("Training finished and model saved.")
                # evaluate
                preds = (model.predict(X_test) > 0.5).astype(int).flatten()
                acc = accuracy_score(y_test, preds)
                st.write(f"Validation accuracy: {acc:.3f}")
                st.write("Classification report:")
                st.text(classification_report(y_test, preds))
                model_loaded = model

# Model prediction on latest setup
st.markdown("## Latest setup prediction")
latest_prediction = None
cross_indices = df[df['cross'].notna()].index.tolist()
if cross_indices:
    latest_cross_idx = cross_indices[-1]
    ret = find_first_retest(df, latest_cross_idx, lookahead=12)
    if ret:
        ret_idx, ret_price, used_ema = ret
        entry_time = df.loc[ret_idx,'date']
        direction = df.loc[latest_cross_idx,'cross']
        st.write(f"Latest retest found at index {ret_idx} date {entry_time} direction {direction}")
        # build sequence ending at ret_idx
        features = ['close','EMA9','EMA15','ema_diff','atr_14','ret1','vol_5','range']
        if ret_idx >= int(lookback)-1:
            seq = df.loc[ret_idx-int(lookback)+1:ret_idx+1, features].values
            if seq.shape[0] == int(lookback):
                if os.path.exists(SCALER_FILE):
                    # attempt to load scaler mean/scale
                    try:
                        mean = np.load(SCALER_FILE)
                        scale = np.load(SCALER_FILE + ".scale")
                        # we used flat scaling earlier; apply same transform
                        flat = seq.reshape(1, -1)
                        flat_scaled = (flat - mean) / scale
                        seq_scaled = flat_scaled.reshape((1, int(lookback), len(features)))
                    except Exception:
                        seq_scaled = seq.reshape((1, int(lookback), len(features)))
                else:
                    seq_scaled = seq.reshape((1, int(lookback), len(features)))
                if model_loaded is not None:
                    prob = float(model_loaded.predict(seq_scaled)[0][0])
                else:
                    prob = 0.5
                latest_prediction = {
                    'entry_time': entry_time,
                    'entry_idx': int(ret_idx),
                    'direction': direction,
                    'entry_price': float(ret_price),
                    'used_ema': used_ema,
                    'prob': prob
                }
                st.write(f"Model probability of TP (sigmoid): {prob:.3f}")
            else:
                st.warning("Sequence not full for prediction (not enough lookback).")
        else:
            st.warning("Not enough history for lookback window for LSTM prediction.")
    else:
        st.info("No recent retest found in the data.")
else:
    st.info("No EMA crosses detected in current data.")

# -------------------- Simulation functions --------------------
def compute_sl_tp_points(entry_price, direction, sl_points, rr_str):
    if rr_str == '1:2':
        tp_points = sl_points * 2.0
    elif rr_str == '1:3':
        tp_points = sl_points * 3.0
    else:
        try:
            factor = float(rr_str)
            tp_points = sl_points * factor
        except Exception:
            tp_points = sl_points * 2.0
    if direction == 'bull':
        sl_price = entry_price - sl_points
        tp_price = entry_price + tp_points
    else:
        sl_price = entry_price + sl_points
        tp_price = entry_price - tp_points
    return sl_price, tp_price, sl_points, tp_points

def simulate_trade_from_entry(df, entry_idx, direction, sl_price, tp_price, max_hold):
    exit_price = None; exit_reason = None; exit_idx = None
    for k in range(entry_idx+1, min(entry_idx + max_hold + 1, len(df))):
        high = float(df.loc[k,'high']); low = float(df.loc[k,'low'])
        if direction == 'bull':
            if high >= tp_price:
                exit_price = tp_price; exit_idx = k; exit_reason = 'TP'; break
            if low <= sl_price:
                exit_price = sl_price; exit_idx = k; exit_reason = 'SL'; break
        else:
            if low <= tp_price:
                exit_price = tp_price; exit_idx = k; exit_reason = 'TP'; break
            if high >= sl_price:
                exit_price = sl_price; exit_idx = k; exit_reason = 'SL'; break
    if exit_price is None:
        exit_idx = min(entry_idx + max_hold, len(df)-1)
        exit_price = float(df.loc[exit_idx,'close'])
        exit_reason = 'Timeout'
    pnl = (exit_price - float(df.loc[entry_idx,'close'])) if direction == 'bull' else (float(df.loc[entry_idx,'close']) - exit_price)
    return exit_idx, exit_price, exit_reason, pnl

# run simulation over all retest setups
if simulate_btn:
    if df.empty:
        st.warning("No data to simulate.")
    else:
        st.info("Simulating retest setups across dataset for selected RR options...")
        new_trades_all = []
        cross_indices = df[df['cross'].notna()].index.tolist()
        for cross_idx in cross_indices:
            ret = find_first_retest(df, cross_idx, lookahead=12)
            if ret is None:
                continue
            ret_idx, ret_price, used_ema = ret
            # skip if sideways at cross moment
            if is_sideways(df.loc[max(0,cross_idx-50):cross_idx+1], atr_window=14, atr_thresh=sideways_atr_ratio):
                continue
            direction = df.loc[cross_idx,'cross']
            # symbol-specific SL/TP rules (BTC special)
            symbol_base = symbol_input.upper()
            # default sl_points provided in sidebar
            sl_points = float(sl_points_default)
            # override for BTC
            if "BTC" in symbol_base:
                # treat given user request: 500 SL points and 1000 TP points
                sl_points = BTC_SL_POINTS
            # simulate for each requested RR
            for rr in rr_options:
                if "BTC" in symbol_base and rr == '1:3':
                    # for BTC user gave 500 and 1000 fixed - but we still support both
                    pass
                sl_price, tp_price, sp, tp_pts = compute_sl_tp_points(ret_price, direction, sl_points, rr)
                exit_idx, exit_price, exit_reason, pnl = simulate_trade_from_entry(df, ret_idx, direction, sl_price, tp_price, int(max_hold))
                trade = {
                    'entry_time': df.loc[ret_idx,'date'].strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_index': int(ret_idx),
                    'exit_time': df.loc[exit_idx,'date'].strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_index': int(exit_idx),
                    'direction': direction,
                    'entry_price': float(ret_price),
                    'exit_price': float(exit_price),
                    'exit_reason': exit_reason,
                    'pnl': float(pnl),
                    'used_ema': used_ema,
                    'rr': rr
                }
                append_trade_csv(trade, rr_choice=rr)
                append_order_log({
                    'time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': symbol_input,
                    'direction': 'BUY' if direction=='bull' else 'SELL',
                    'entry_price': float(ret_price),
                    'sl': sl_price,
                    'tp': tp_price,
                    'lot': 0.01,
                    'event': 'SIM_ENTRY',
                    'note': rr
                })
                new_trades_all.append(trade)
        st.success(f"Simulated {len(new_trades_all)} trades. Files updated: 1:2 and/or 1:3 CSVs.")

# Download / clear functions
if download_trades_btn:
    if os.path.exists(TRADES_CSV_1_2):
        with open(TRADES_CSV_1_2, "rb") as f:
            st.download_button("Download 1:2 trades", f, file_name=os.path.basename(TRADES_CSV_1_2))
    else:
        st.warning("No 1:2 trades file available.")

if download_trades_1_3_btn:
    if os.path.exists(TRADES_CSV_1_3):
        with open(TRADES_CSV_1_3, "rb") as f:
            st.download_button("Download 1:3 trades", f, file_name=os.path.basename(TRADES_CSV_1_3))
    else:
        st.warning("No 1:3 trades file available.")

if clear_trades_btn:
    if os.path.exists(TRADES_CSV_1_2):
        os.remove(TRADES_CSV_1_2)
    if os.path.exists(TRADES_CSV_1_3):
        os.remove(TRADES_CSV_1_3)
    if os.path.exists(ORDERS_CSV):
        os.remove(ORDERS_CSV)
    ensure_csv_with_header(TRADES_CSV_1_2, ["entry_time","entry_index","exit_time","exit_index","direction","entry_price","exit_price","exit_reason","pnl","used_ema","rr"])
    ensure_csv_with_header(TRADES_CSV_1_3, ["entry_time","entry_index","exit_time","exit_index","direction","entry_price","exit_price","exit_reason","pnl","used_ema","rr"])
    ensure_csv_with_header(ORDERS_CSV, ["time","symbol","direction","entry_price","sl","tp","lot","event","note"])
    st.success("All trade & order CSVs cleared.")

# ---------------- Trade history (table) and focused chart below ----------------
st.markdown("## Trade history and focused chart")
# show combined view of both rr CSVs
trades_12 = load_trades('1:2')
trades_13 = load_trades('1:3')
combined_trades = pd.concat([trades_12, trades_13], ignore_index=True) if (not trades_12.empty or not trades_13.empty) else pd.DataFrame()
if combined_trades.empty:
    st.info("No simulated trades yet. Run simulation to generate trades.")
else:
    combined_trades = combined_trades.sort_values('entry_time', ascending=False).reset_index(drop=True)
    st.dataframe(combined_trades[['entry_time','direction','entry_price','exit_price','exit_reason','pnl','rr']].head(200))
    # selection via selectbox picking entry_time string
    choice = st.selectbox("Select trade (entry_time) to view focused chart", options=combined_trades['entry_time'].tolist())
    sel_row = combined_trades[combined_trades['entry_time'] == choice].iloc[0]
    st.markdown("### Selected trade details")
    st.write(f"Direction: **{sel_row['direction'].upper()}**, RR: {sel_row.get('rr','N/A')}")
    st.write(f"Entry time: {sel_row['entry_time']}, Entry price: {sel_row['entry_price']:.3f}")
    st.write(f"Exit time: {sel_row['exit_time']}, Exit price: {sel_row['exit_price']:.3f} ({sel_row['exit_reason']})")
    st.write(f"P&L: {sel_row['pnl']:.3f}")

    # focused chart (below the main chart)
    try:
        entry_dt = pd.to_datetime(sel_row['entry_time'])
        # find nearest index in df
        diffs = (df['date'] - entry_dt).abs()
        entry_idx = diffs.idxmin()
        # allow slider to choose candles either side
        focus_candles = st.slider("Candles around entry (each side)", min_value=5, max_value=200, value=40)
        start_idx = max(0, entry_idx - focus_candles)
        end_idx = min(len(df)-1, entry_idx + focus_candles)
        focus_df = df.iloc[start_idx:end_idx+1].copy().reset_index(drop=True)
        ffig = go.Figure()
        ffig.add_trace(go.Candlestick(x=focus_df['date'], open=focus_df['open'], high=focus_df['high'], low=focus_df['low'], close=focus_df['close'], name='price'))
        ffig.add_trace(go.Scatter(x=focus_df['date'], y=focus_df['EMA9'], mode='lines', name='EMA9'))
        ffig.add_trace(go.Scatter(x=focus_df['date'], y=focus_df['EMA15'], mode='lines', name='EMA15'))
        # entry/exit markers
        ffig.add_trace(go.Scatter(x=[pd.to_datetime(sel_row['entry_time'])], y=[sel_row['entry_price']], mode='markers+text', marker=dict(color='green', size=12, symbol='triangle-up'), text=["Entry"], textposition="top center"))
        ffig.add_trace(go.Scatter(x=[pd.to_datetime(sel_row['exit_time'])], y=[sel_row['exit_price']], mode='markers+text', marker=dict(color='red', size=12, symbol='triangle-down'), text=["Exit"], textposition="bottom center"))
        ffig.update_layout(height=420, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(ffig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to draw focused chart: {e}")

# ---------------- Performance summary ----------------
st.markdown("## Performance summary")
trades_all = pd.concat([trades_12, trades_13], ignore_index=True) if (not trades_12.empty or not trades_13.empty) else pd.DataFrame()
if trades_all.empty:
    st.info("No trades to summarize.")
else:
    total_pnl = trades_all['pnl'].sum()
    wins = trades_all[trades_all['pnl'] > 0].shape[0]
    losses = trades_all[trades_all['pnl'] <= 0].shape[0]
    avg_pnl = trades_all['pnl'].mean()
    st.metric("Total PnL", f"{total_pnl:.2f}")
    c1,c2,c3 = st.columns(3)
    c1.metric("Trades", trades_all.shape[0])
    c2.metric("Wins", wins)
    c3.metric("Avg PnL", f"{avg_pnl:.2f}")

# ---------------- Manual model test (optional) ----------------
st.markdown("## Manual LSTM test")
if model_loaded is None:
    st.info("No LSTM model available to test.")
else:
    idx_choice = st.number_input("Pick candle index to run model on (0 = oldest)", min_value=0, max_value=max(0, len(df)-1), value=max(0, len(df)-1))
    if st.button("Run model prediction for chosen index"):
        try:
            features = ['close','EMA9','EMA15','ema_diff','atr_14','ret1','vol_5','range']
            if idx_choice < int(lookback)-1:
                st.warning("Index too small for lookback.")
            else:
                seq = df.loc[int(idx_choice)-int(lookback)+1:int(idx_choice)+1, features].values
                flat = seq.reshape(1, -1)
                if os.path.exists(SCALER_FILE):
                    try:
                        mean = np.load(SCALER_FILE)
                        scale = np.load(SCALER_FILE + ".scale")
                        flat_scaled = (flat - mean) / scale
                        seq_scaled = flat_scaled.reshape((1, int(lookback), len(features)))
                    except Exception:
                        seq_scaled = flat.reshape((1, int(lookback), len(features)))
                else:
                    seq_scaled = flat.reshape((1, int(lookback), len(features)))
                p = float(model_loaded.predict(seq_scaled)[0][0])
                st.write(f"Model prob of TP: {p:.3f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("---")
st.markdown("**Notes:**\n- This app simulates trades using historical Finnhub candles and an LSTM model. It does NOT execute real orders.\n- For production algo trading, you must add a separate execution agent with proper risk management, slippage handling, position sizing, and exchange APIs.")

# End of app
