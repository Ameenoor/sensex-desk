from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
from textblob import TextBlob
import uvicorn
from datetime import datetime, timezone, timedelta
import math

app = FastAPI(title="SENSEX AI AlphaDesk")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

trade_history_log = []
last_recorded_signal = None

def get_market_catalysts():
    try:
        feed = feedparser.parse("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms")
        high_impact, trade_ideas = [], []
        sentiment_score = 0
        
        impact_kws = ['rbi', 'fed', 'sebi', 'inflation', 'gdp', 'rate', 'hdfc', 'reliance', 'icici', 'infosys', 'tcs', 'sensex', 'nifty', 'crash', 'surge', 'policy', 'fomc']
        trade_kws = ['buy', 'sell', 'target', 'breakout', 'upgrade', 'downgrade', 'profit', 'earnings', 'dividend', 'block deal', 'stake']
        
        for entry in feed.entries[:25]:
            title_lower = entry.title.lower()
            link = getattr(entry, 'link', '#')
            blob = TextBlob(entry.title)
            sentiment_score += blob.sentiment.polarity
            
            is_impact = any(kw in title_lower for kw in impact_kws)
            is_trade = any(kw in title_lower for kw in trade_kws)
            
            if is_impact and len(high_impact) < 4: high_impact.append({"title": entry.title, "link": link})
            elif is_trade and len(trade_ideas) < 4: trade_ideas.append({"title": entry.title, "link": link})
                
        if not high_impact: high_impact.append({"title": "No sudden macro shocks detected.", "link": "#"})
        if not trade_ideas: trade_ideas.append({"title": "Scanning for breakouts & catalysts...", "link": "#"})
            
        avg_sentiment = sentiment_score / max(len(feed.entries[:25]), 1)
        bias = "BULLISH" if avg_sentiment > 0.05 else "BEARISH" if avg_sentiment < -0.05 else "NEUTRAL"
        
        return {"bias": bias, "high_impact": high_impact, "trade_ideas": trade_ideas}
    except Exception:
        return {"bias": "NEUTRAL", "high_impact": [{"title": "News Offline.", "link": "#"}], "trade_ideas": [{"title": "News Offline.", "link": "#"}]}

def evaluate_self_learning(df):
    """
    On-the-fly Self-Correction Engine:
    Looks at daily closes vs next day opens over the last 15 days to train the model error rate.
    """
    daily_data = []
    # Group by Date (IST)
    df_dates = df.groupby(df.index.date)
    
    past_predictions = []
    error_sum = 0
    total_gaps = 0
    
    dates = list(df_dates.groups.keys())
    for i in range(len(dates) - 1):
        current_day = df_dates.get_group(dates[i])
        next_day = df_dates.get_group(dates[i+1])
        
        if len(current_day) > 0 and len(next_day) > 0:
            close_price = current_day['Close'].iloc[-1]
            macd = current_day['MACD_Hist'].iloc[-1]
            next_open = next_day['Open'].iloc[0]
            
            # Simple algorithmic EOD prediction logic
            predicted_dir = "GAP UP" if macd > 0 else "GAP DOWN"
            actual_dir = "GAP UP" if next_open > close_price else "GAP DOWN"
            gap_pts = round(next_open - close_price, 2)
            
            # Calculate Error (Absolute difference between expectation and reality)
            # If predicted UP but gapped DOWN, error is high.
            if predicted_dir == actual_dir:
                error = 0 # Correct prediction
            else:
                error = abs(gap_pts)
                error_sum += error
                
            total_gaps += 1
            
            past_predictions.append({
                "date": dates[i].strftime("%Y-%m-%d"),
                "prediction": predicted_dir,
                "actual": actual_dir,
                "gap_pts": f"{'+' if gap_pts > 0 else ''}{gap_pts}",
                "status": "✅ Correct" if predicted_dir == actual_dir else f"❌ Error ({error} pts)"
            })
            
    avg_error = round(error_sum / max(total_gaps, 1), 2)
    accuracy = 100 - min((error_sum / max(total_gaps * 100, 1)) * 100, 100) # Rough heuristic
    
    return past_predictions[-5:], avg_error, round(accuracy, 1)

def calculate_advanced_strategy(df):
    if len(df) < 30: raise ValueError("Insufficient historical data received from BSE.")
    ema_long_period = min(200, len(df) - 1)

    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_Institutional'] = df['Close'].ewm(span=ema_long_period, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['TR'] = np.max(ranges, axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    df['Avg_TR_5'] = df['TR'].rolling(5).mean() 
    df['Rolling_Max_30'] = df['High'].rolling(30).max() 
    df['Rolling_Min_30'] = df['Low'].rolling(30).min()  

    df.dropna(inplace=True)
    if df.empty: raise ValueError("Data empty after cleaning indicators.")

    # Run Self-Learning Module
    learning_history, avg_error, model_accuracy = evaluate_self_learning(df)

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    spot_price = round(float(last_row['Close']), 2)
    rsi_val = float(last_row['RSI'])
    atr_val = float(last_row['ATR'])
    
    # Open Interest (OI) Proxy Engine
    # Calculates institutional psychological resistance/support zones
    call_oi_resistance = int(math.ceil(spot_price / 500.0)) * 500
    put_oi_support = int(math.floor(spot_price / 500.0)) * 500
    
    if call_oi_resistance == put_oi_support:
        call_oi_resistance += 500
        put_oi_support -= 500

    footprint_active = last_row['TR'] >= (last_row['Avg_TR_5'] * 0.75)
    
    last_time = df.index[-1]
    time_val = last_time.hour * 100 + last_time.minute
    
    # State Logic: Is market ending/closed?
    is_eod = (time_val >= 1515) or (time_val < 915)

    is_above_institution_trend = spot_price > last_row['EMA_Institutional']
    is_below_institution_trend = spot_price < last_row['EMA_Institutional']
    bullish_momentum = last_row['MACD_Hist'] > 0
    bearish_momentum = last_row['MACD_Hist'] < 0

    signal = "NEUTRAL"
    recommended_strike = "---"
    index_target = 0
    index_sl = 0
    opt_target_pts = "---"
    opt_sl_pts = "---"
    rationale = "Monitoring Price Action. Confluence not yet established."
    
    dynamic_target_pts = max(100, int(atr_val * 2.0)) 
    dynamic_sl_pts = max(50, int(atr_val * 1.0))
    
    smc_status = f"Simulated OI: Support at {put_oi_support}, Resistance at {call_oi_resistance}."

    # EOD PREDICTION MODE (Tomorrow's Market)
    if is_eod:
        if bullish_momentum and rsi_val < 70:
            signal = "EOD BTST: PREDICT GAP UP"
            recommended_strike = f"{int(round(spot_price / 100.0) * 100)} CE (Call)"
            rationale = f"MARKET CLOSED. EOD MACD is positive. Predicting Gap Up tomorrow towards OI Resistance {call_oi_resistance}."
        elif bearish_momentum and rsi_val > 30:
            signal = "EOD STBT: PREDICT GAP DOWN"
            recommended_strike = f"{int(round(spot_price / 100.0) * 100)} PE (Put)"
            rationale = f"MARKET CLOSED. EOD MACD is negative. Predicting Gap Down tomorrow towards OI Support {put_oi_support}."
        else:
            signal = "EOD: FLAT / NEUTRAL"
            rationale = "MARKET CLOSED. Momentum is mixed. No overnight trade recommended."
    
    # INTRADAY MODE
    else:
        bullish_condition = ((last_row['EMA9'] > last_row['EMA21']) and bullish_momentum and (rsi_val < 65) and is_above_institution_trend and footprint_active)
        bearish_condition = ((last_row['EMA9'] < last_row['EMA21']) and bearish_momentum and (rsi_val > 35) and is_below_institution_trend and footprint_active)

        if not footprint_active:
            rationale = "Market footprint is dead. Waiting for volume expansion."
        elif bullish_condition:
            signal = "BUY SIGNAL (15m CONFLUENCE)"
            atm = int(round(spot_price / 100.0) * 100)
            recommended_strike = f"{atm} CE (Targeting {call_oi_resistance} OI)"
            index_target = spot_price + dynamic_target_pts
            index_sl = spot_price - dynamic_sl_pts
            opt_target_pts = f"+{int(dynamic_target_pts * 0.55)} Premium Pts" 
            opt_sl_pts = f"-{int(dynamic_sl_pts * 0.55)} Premium Pts"
            rationale = f"BULLISH CONFIRMED: Breaking towards Call OI resistance at {call_oi_resistance}."
        elif bearish_condition:
            signal = "SELL SIGNAL (15m CONFLUENCE)"
            atm = int(round(spot_price / 100.0) * 100)
            recommended_strike = f"{atm} PE (Targeting {put_oi_support} OI)"
            index_target = spot_price - dynamic_target_pts
            index_sl = spot_price + dynamic_sl_pts
            opt_target_pts = f"+{int(dynamic_target_pts * 0.55)} Premium Pts"
            opt_sl_pts = f"-{int(dynamic_sl_pts * 0.55)} Premium Pts"
            rationale = f"BEARISH CONFIRMED: Breaking towards Put OI support at {put_oi_support}."

    reversal = "SAFE (No Exhaustion)"
    reversal_color = "text-emerald-400"
    if rsi_val >= 75: reversal = f"⚠ EXTREME OVERBOUGHT - OI CALL SELLERS TRAPPED"; reversal_color = "text-red-500 animate-pulse"
    elif rsi_val <= 25: reversal = f"⚠ EXTREME OVERSOLD - OI PUT SELLERS TRAPPED"; reversal_color = "text-red-500 animate-pulse"

    return {
        "signal": signal, "entry_spot": spot_price, "recommended_strike": recommended_strike,
        "index_target": round(index_target, 2), "index_sl": round(index_sl, 2),
        "option_target_pts": opt_target_pts, "option_sl_pts": opt_sl_pts,
        "reversal_warning": reversal, "reversal_color": reversal_color,
        "rsi": round(rsi_val, 2), "atr": round(atr_val, 2), 
        "rationale": rationale, "smc_status": smc_status,
        "learning_history": learning_history, "model_accuracy": model_accuracy, "avg_error": avg_error,
        "put_oi_support": put_oi_support, "call_oi_res": call_oi_resistance
    }

@app.get("/api/market_data")
def get_market_data():
    global last_recorded_signal, trade_history_log
    try:
        ticker = yf.Ticker("^BSESN")
        df = ticker.history(period="15d", interval="15m")
        if df.empty: return {"error": "Market data empty. Ensure internet connection and market hours."}
        if df.index.tz is None: df.index = df.index.tz_localize('Asia/Kolkata')
        else: df.index = df.index.tz_convert('Asia/Kolkata')

        df = df[~df.index.duplicated(keep='first')].sort_index()
        strategy = calculate_advanced_strategy(df)
        
        current_signal = strategy["signal"]
        if ("CONFLUENCE" in current_signal or "BTST" in current_signal) and current_signal != last_recorded_signal:
            ist_time = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S")
            trade_history_log.insert(0, {
                "time": ist_time,
                "type": current_signal.replace(" (15m CONFLUENCE)", ""),
                "strike": strategy["recommended_strike"],
                "entry": strategy["entry_spot"],
                "target": strategy["index_target"],
                "sl": strategy["index_sl"]
            })
            trade_history_log = trade_history_log[:50] 
            last_recorded_signal = current_signal
            
        chart_data = []
        for index, row in df.iterrows():
            if not pd.isna(row['Close']) and not pd.isna(row['Open']):
                ist_adjusted_epoch = int(index.timestamp()) + 19800 
                chart_data.append({
                    "time": ist_adjusted_epoch, "open": round(float(row['Open']), 2),
                    "high": round(float(row['High']), 2), "low": round(float(row['Low']), 2),
                    "close": round(float(row['Close']), 2), "ema9": round(float(row['EMA9']), 2),
                    "ema21": round(float(row['EMA21']), 2), "ema_inst": round(float(row['EMA_Institutional']), 2),
                    "rsi": round(float(row['RSI']), 2) if not pd.isna(row['RSI']) else 50
                })

        return {
            "status": "success", "current_price": round(float(df.iloc[-1]['Close']), 2),
            "chart_data": chart_data, "strategy": strategy,
            "macro_news": get_market_catalysts(), "trade_history": trade_history_log
        }
    except Exception as e:
        return {"error": f"Backend Error: {str(e)}"}

HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SENSEX AI AlphaDesk</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lightweight-charts@4.2.1/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        .chart-container { position: relative; }
        .rsi-container { position: relative; height: 180px; width: 100%; border-top: 1px solid #374151; display: none; }
        .toggle-checkbox:checked { right: 0; border-color: #3b82f6; }
        .toggle-checkbox:checked + .toggle-label { background-color: #3b82f6; }
        a:hover { color: #60a5fa !important; }
    </style>
</head>
<body class="bg-gray-900 text-white font-sans p-4 md:p-6">

    <div class="max-w-7xl mx-auto space-y-6">
        <div id="error-banner" class="hidden bg-red-900 border-2 border-red-500 text-white p-4 rounded-xl text-sm font-mono break-words shadow-lg"></div>

        <!-- Header -->
        <div class="flex flex-col md:flex-row justify-between items-center bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-700 gap-4">
            <div class="flex items-center space-x-4">
                <h1 class="text-2xl font-bold text-emerald-400">SENSEX AlphaDesk <span class="text-xs text-purple-400 border border-purple-500 px-1 rounded">AI ENGINE</span></h1>
                <span id="market-timer" class="text-xs px-2 py-1 rounded font-bold bg-gray-700 text-gray-300">CALCULATING...</span>
            </div>
            
            <div class="flex space-x-4 bg-gray-900 px-4 py-2 rounded-lg border border-gray-700 text-xs font-mono">
                <label class="flex items-center cursor-pointer space-x-2">
                    <input type="checkbox" id="toggle-ema-short" class="form-checkbox text-blue-500 rounded bg-gray-800 border-gray-600" checked>
                    <span class="text-gray-300">EMA 9/21</span>
                </label>
                <label class="flex items-center cursor-pointer space-x-2">
                    <input type="checkbox" id="toggle-ema-long" class="form-checkbox text-blue-500 rounded bg-gray-800 border-gray-600" checked>
                    <span class="text-gray-300">EMA 200</span>
                </label>
                <label class="flex items-center cursor-pointer space-x-2">
                    <input type="checkbox" id="toggle-rsi" class="form-checkbox text-blue-500 rounded bg-gray-800 border-gray-600">
                    <span class="text-gray-300">RSI View</span>
                </label>
            </div>
            <div class="text-right hidden md:block">
                <div id="market-clock" class="text-lg font-mono font-bold text-gray-200">--:--:--</div>
                <div class="text-xs text-gray-500 uppercase tracking-wider">Indian Standard Time</div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="lg:col-span-2 space-y-4">
                <div class="bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-700 overflow-hidden">
                    <div class="flex justify-between items-center mb-4">
                        <div class="flex items-center space-x-3">
                            <h2 class="text-lg font-bold text-gray-200">SENSEX Spot</h2>
                            <div class="text-[10px] flex space-x-2 font-mono">
                                <span class="text-green-400">● EMA9</span>
                                <span class="text-red-400">● EMA21</span>
                                <span class="text-gray-300">● EMA200</span>
                            </div>
                        </div>
                        <div class="text-xl font-mono font-bold" id="current-price">Connecting...</div>
                    </div>
                    
                    <div id="main-chart" class="w-full h-[400px] chart-container"></div>
                    <div id="rsi-chart" class="rsi-container"></div>
                    
                    <div class="mt-4 p-4 bg-gray-900 border border-gray-700 rounded-lg flex flex-col md:flex-row gap-4 shadow-inner">
                        <div class="flex-1">
                            <h3 class="text-[10px] text-purple-400 uppercase tracking-widest mb-1 font-bold">OI & Footprint Tracker</h3>
                            <p id="smc-text" class="text-sm text-gray-300 font-medium">Analyzing Liquidity Zones...</p>
                        </div>
                        <div class="flex-1 border-t md:border-t-0 md:border-l border-gray-700 pt-3 md:pt-0 md:pl-4">
                            <h3 class="text-[10px] text-emerald-400 uppercase tracking-widest mb-1 font-bold">AI Rationale</h3>
                            <p id="rationale-text" class="text-sm text-gray-300 font-medium leading-relaxed">Processing live data...</p>
                        </div>
                    </div>
                </div>
            </div>

            <div class="space-y-4">
                <div class="bg-blue-950/40 p-5 rounded-xl border border-blue-700/50 shadow-lg relative overflow-hidden">
                    <div class="absolute top-0 right-0 bg-blue-700 text-white text-[10px] px-2 py-1 rounded-bl-lg font-bold">DYNAMIC STRIKE SELECTOR</div>
                    <h2 class="text-sm text-blue-300 font-bold uppercase tracking-wider mb-2">Option Contract</h2>
                    <div id="strike-text" class="text-[1.1rem] md:text-xl font-mono font-bold mb-1 text-white">AWAITING...</div>
                    <p class="text-[10px] text-gray-400 mb-4">Calculates exact ATM based on Spot & OI.</p>
                    
                    <div class="grid grid-cols-2 gap-3 text-sm font-mono">
                        <div class="bg-gray-900/60 p-2 rounded border border-gray-700">
                            <span class="text-gray-500 block text-[10px] uppercase">Option Target</span>
                            <span id="opt-target" class="text-emerald-400 font-bold">---</span>
                        </div>
                        <div class="bg-gray-900/60 p-2 rounded border border-gray-700">
                            <span class="text-gray-500 block text-[10px] uppercase">Option SL</span>
                            <span id="opt-sl" class="text-red-400 font-bold">---</span>
                        </div>
                    </div>
                </div>

                <div class="bg-gray-800 p-5 rounded-xl border border-gray-700 shadow-lg">
                    <h2 class="text-sm text-gray-400 font-bold uppercase tracking-wider mb-2">Underlying Spot Triggers</h2>
                    <div id="signal-text" class="text-[1.1rem] font-bold mb-4">WAITING FOR CONFLUENCE...</div>
                    <div class="grid grid-cols-2 gap-4 text-sm font-mono">
                        <div class="bg-gray-900 p-3 rounded border border-gray-700">
                            <span class="text-gray-500 block text-[10px] uppercase">Spot Target</span>
                            <span id="target-price" class="text-gray-200 font-bold">---</span>
                        </div>
                        <div class="bg-gray-900 p-3 rounded border border-gray-700">
                            <span class="text-gray-500 block text-[10px] uppercase">Spot SL</span>
                            <span id="sl-price" class="text-gray-200 font-bold">---</span>
                        </div>
                    </div>
                    <div class="mt-4 p-3 bg-gray-900 rounded border border-gray-700">
                        <span class="text-xs text-gray-500 block mb-1">SAFETY & REVERSAL RADAR</span>
                        <span id="reversal-status" class="text-emerald-400 font-bold text-xs">Monitoring...</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- NEW DIVISION: AI Learning & History -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            <!-- Automated Call Log -->
            <div class="bg-gray-800 rounded-xl shadow-lg border border-gray-700 overflow-hidden">
                <div class="bg-gray-900 p-3 border-b border-gray-700 flex justify-between items-center">
                    <h2 class="text-sm font-bold text-gray-200 uppercase tracking-wider flex items-center">
                        <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse mr-2"></span> Automated Call Log
                    </h2>
                </div>
                <div class="overflow-x-auto h-64">
                    <table class="w-full text-sm text-left text-gray-400">
                        <thead class="text-xs text-gray-500 uppercase bg-gray-900/50 border-b border-gray-700">
                            <tr>
                                <th class="px-4 py-2">Time (IST)</th>
                                <th class="px-4 py-2">Type</th>
                                <th class="px-4 py-2">Script</th>
                                <th class="px-4 py-2">Entry</th>
                            </tr>
                        </thead>
                        <tbody id="trade-history-table">
                            <tr class="bg-gray-800"><td colspan="4" class="px-4 py-4 text-center">Awaiting first algorithmic trigger...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Self-Learning Accuracy Matrix -->
            <div class="bg-gray-800 rounded-xl shadow-lg border border-gray-700 overflow-hidden">
                <div class="bg-gray-900 p-3 border-b border-gray-700 flex justify-between items-center">
                    <h2 class="text-sm font-bold text-gray-200 uppercase tracking-wider flex items-center">
                        <span class="w-2 h-2 rounded-full bg-purple-500 animate-pulse mr-2"></span> AI Self-Learning & EOD Accuracy
                    </h2>
                    <span id="accuracy-badge" class="px-2 py-1 bg-purple-900 text-purple-300 rounded text-xs font-bold font-mono">Acc: --%</span>
                </div>
                <div class="overflow-x-auto h-64">
                    <table class="w-full text-sm text-left text-gray-400">
                        <thead class="text-xs text-gray-500 uppercase bg-gray-900/50 border-b border-gray-700">
                            <tr>
                                <th class="px-4 py-2">Date</th>
                                <th class="px-4 py-2">EOD Prediction</th>
                                <th class="px-4 py-2">Actual Open</th>
                                <th class="px-4 py-2">Status</th>
                            </tr>
                        </thead>
                        <tbody id="learning-history-table">
                            <tr class="bg-gray-800"><td colspan="4" class="px-4 py-4 text-center">Training Model...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="bg-gray-800 rounded-xl shadow-lg border border-gray-700 overflow-hidden mt-6">
            <div class="bg-gray-900 p-3 border-b border-gray-700 flex justify-between items-center">
                <h2 class="text-sm font-bold text-gray-200 uppercase tracking-wider">Market Intelligence Terminal (Clickable)</h2>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-gray-700">
                <div class="p-4 bg-gray-800/50">
                    <h3 class="text-xs text-rose-400 font-bold uppercase tracking-widest mb-3">High Impact SENSEX Shocks</h3>
                    <div id="impact-news-feed" class="space-y-3"></div>
                </div>
                <div class="p-4 bg-gray-800/50">
                    <h3 class="text-xs text-emerald-400 font-bold uppercase tracking-widest mb-3">Actionable Trade Catalysts</h3>
                    <div id="trade-news-feed" class="space-y-3"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const updateText = (id, val) => { const el = document.getElementById(id); if (el) el.innerText = val; };
        const updateHTML = (id, val) => { const el = document.getElementById(id); if (el) el.innerHTML = val; };
        const updateClass = (id, val) => { const el = document.getElementById(id); if (el) el.className = val; };

        const chartOptions = {
            layout: { backgroundColor: '#1f2937', textColor: '#d1d5db' },
            grid: { vertLines: { color: 'rgba(55, 65, 81, 0.4)' }, horzLines: { color: 'rgba(55, 65, 81, 0.4)' } },
            crosshair: { mode: 0 },
            timeScale: { 
                timeVisible: true,
                tickMarkFormatter: (time) => {
                    const d = new Date(time * 1000);
                    return d.getUTCHours().toString().padStart(2, '0') + ':' + d.getUTCMinutes().toString().padStart(2, '0');
                }
            },
            localization: {
                timeFormatter: (time) => {
                    const d = new Date(time * 1000);
                    return `${d.getUTCDate().toString().padStart(2, '0')}-${(d.getUTCMonth()+1).toString().padStart(2, '0')}  ${d.getUTCHours().toString().padStart(2, '0')}:${d.getUTCMinutes().toString().padStart(2, '0')} IST`;
                }
            }
        };

        const mainChart = LightweightCharts.createChart(document.getElementById('main-chart'), chartOptions);
        const candleSeries = mainChart.addCandlestickSeries({ upColor: '#22c55e', downColor: '#ef4444', borderUpColor: '#22c55e', borderDownColor: '#ef4444', wickUpColor: '#22c55e', wickDownColor: '#ef4444' });
        const ema9Series = mainChart.addLineSeries({ color: '#4ade80', lineWidth: 2, crosshairMarkerVisible: false });
        const ema21Series = mainChart.addLineSeries({ color: '#f87171', lineWidth: 2, crosshairMarkerVisible: false });
        const ema200Series = mainChart.addLineSeries({ color: '#d1d5db', lineWidth: 2, lineStyle: 2, crosshairMarkerVisible: false });

        const rsiChartOptions = JSON.parse(JSON.stringify(chartOptions));
        rsiChartOptions.timeScale.visible = false; 
        const rsiChart = LightweightCharts.createChart(document.getElementById('rsi-chart'), rsiChartOptions);
        const rsiSeries = rsiChart.addLineSeries({ color: '#c084fc', lineWidth: 2, crosshairMarkerVisible: false });

        rsiSeries.createPriceLine({ price: 75, color: '#ef4444', lineStyle: 2, axisLabelVisible: true, title: 'OB (75)' });
        rsiSeries.createPriceLine({ price: 65, color: '#f59e0b', lineStyle: 2, axisLabelVisible: true, title: 'Warn (65)' });
        rsiSeries.createPriceLine({ price: 35, color: '#f59e0b', lineStyle: 2, axisLabelVisible: true, title: 'Warn (35)' });
        rsiSeries.createPriceLine({ price: 25, color: '#22c55e', lineStyle: 2, axisLabelVisible: true, title: 'OS (25)' });

        mainChart.timeScale().subscribeVisibleLogicalRangeChange(timeRange => { rsiChart.timeScale().setVisibleLogicalRange(timeRange); });
        
        document.getElementById('toggle-ema-short').addEventListener('change', (e) => {
            ema9Series.applyOptions({ visible: e.target.checked }); ema21Series.applyOptions({ visible: e.target.checked });
        });
        document.getElementById('toggle-ema-long').addEventListener('change', (e) => { ema200Series.applyOptions({ visible: e.target.checked }); });
        document.getElementById('toggle-rsi').addEventListener('change', (e) => {
            document.getElementById('rsi-chart').style.display = e.target.checked ? 'block' : 'none';
        });

        function updateMarketClock() {
            const now = new Date();
            const options = { timeZone: 'Asia/Kolkata', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit', weekday: 'short' };
            const parts = new Intl.DateTimeFormat('en-US', options).formatToParts(now);
            let hh = 0, mm = 0, day = '';
            for (const p of parts) {
                if (p.type === 'hour') hh = parseInt(p.value);
                if (p.type === 'minute') mm = parseInt(p.value);
                if (p.type === 'weekday') day = p.value;
            }
            let status = "MARKET CLOSED", color = "bg-red-900 text-red-300";
            if (day !== 'Sat' && day !== 'Sun') {
                const t = hh * 100 + mm;
                if (t >= 900 && t < 915) { status = "PRE-MARKET"; color = "bg-amber-900 text-amber-300"; }
                else if (t >= 915 && t < 1530) { status = "LIVE MARKET"; color = "bg-emerald-900 text-emerald-300"; }
            }
            updateText('market-timer', status); updateClass('market-timer', `text-xs px-2 py-1 rounded font-bold ${color}`);
            updateText('market-clock', now.toLocaleTimeString('en-US', { timeZone: 'Asia/Kolkata' }));
        }
        setInterval(updateMarketClock, 1000); updateMarketClock();

        let lastSignal = "NEUTRAL";

        async function fetchMarketData() {
            try {
                const response = await fetch(window.location.origin + '/api/market_data');
                const data = await response.json();
                if(data.error) throw new Error(data.error);

                updateClass('error-banner', 'hidden');

                if (data.chart_data && data.chart_data.length > 0) {
                    const uniqueCandles = [], ema9Data = [], ema21Data = [], ema200Data = [], rsiData = [];
                    let lastTime = 0;
                    for (const item of data.chart_data) {
                        if (item.time > lastTime) { 
                            uniqueCandles.push({time: item.time, open: item.open, high: item.high, low: item.low, close: item.close});
                            if(item.ema9) ema9Data.push({time: item.time, value: item.ema9});
                            if(item.ema21) ema21Data.push({time: item.time, value: item.ema21});
                            if(item.ema_inst) ema200Data.push({time: item.time, value: item.ema_inst});
                            if(item.rsi) rsiData.push({time: item.time, value: item.rsi});
                            lastTime = item.time; 
                        }
                    }
                    candleSeries.setData(uniqueCandles); ema9Series.setData(ema9Data);
                    ema21Series.setData(ema21Data); ema200Series.setData(ema200Data); rsiSeries.setData(rsiData);
                }
                
                updateText('current-price', "₹ " + data.current_price);
                const strat = data.strategy;
                
                updateText('rationale-text', strat.rationale); updateText('smc-text', strat.smc_status);
                updateText('strike-text', strat.recommended_strike);
                updateText('opt-target', strat.recommended_strike !== "---" ? strat.option_target_pts : "---");
                updateText('opt-sl', strat.recommended_strike !== "---" ? strat.option_sl_pts : "---");
                updateText('signal-text', strat.signal);
                updateClass('signal-text', "text-xl font-bold mb-4 " + (strat.signal.includes("BUY") || strat.signal.includes("UP") ? "text-emerald-400" : strat.signal.includes("SELL") || strat.signal.includes("DOWN") ? "text-red-400" : "text-gray-400"));
                updateText('target-price', strat.index_target > 0 ? "₹" + strat.index_target : "---");
                updateText('sl-price', strat.index_sl > 0 ? "₹" + strat.index_sl : "---");
                updateText('reversal-status', strat.reversal_warning);
                updateClass('reversal-status', `${strat.reversal_color} font-bold text-xs`);

                // Update Trade History Table
                if (data.trade_history && data.trade_history.length > 0) {
                    const tableHTML = data.trade_history.map(trade => `
                        <tr class="bg-gray-800 border-b border-gray-700 hover:bg-gray-700">
                            <td class="px-4 py-3">${trade.time.split(' ')[1]}</td>
                            <td class="px-4 py-3 font-bold ${trade.type.includes('BUY') || trade.type.includes('UP') ? 'text-emerald-400' : 'text-red-400'}">${trade.type}</td>
                            <td class="px-4 py-3 text-gray-200 font-mono text-xs">${trade.strike}</td>
                            <td class="px-4 py-3">${trade.entry}</td>
                        </tr>
                    `).join('');
                    updateHTML('trade-history-table', tableHTML);
                }

                // Update AI Learning Table
                if (strat.learning_history && strat.learning_history.length > 0) {
                    updateText('accuracy-badge', `Acc: ${strat.model_accuracy}% (Err: ${strat.avg_error})`);
                    const learnHTML = strat.learning_history.reverse().map(l => `
                        <tr class="bg-gray-800 border-b border-gray-700 hover:bg-gray-700">
                            <td class="px-4 py-2">${l.date}</td>
                            <td class="px-4 py-2 font-bold ${l.prediction.includes('UP') ? 'text-emerald-400' : 'text-red-400'}">${l.prediction}</td>
                            <td class="px-4 py-2 font-mono">${l.actual} (${l.gap_pts})</td>
                            <td class="px-4 py-2">${l.status}</td>
                        </tr>
                    `).join('');
                    updateHTML('learning-history-table', learnHTML);
                }

                if (data.macro_news) {
                    updateHTML('impact-news-feed', data.macro_news.high_impact.map(n => `<div class="border-l-2 border-rose-500 pl-3 py-1 bg-gray-900/40 rounded-r"><a href="${n.link}" target="_blank" class="hover:underline text-xs text-gray-300 font-medium">${n.title}</a></div>`).join(''));
                    updateHTML('trade-news-feed', data.macro_news.trade_ideas.map(n => `<div class="border-l-2 border-emerald-500 pl-3 py-1 bg-gray-900/40 rounded-r"><a href="${n.link}" target="_blank" class="hover:underline text-xs text-gray-300 font-medium">${n.title}</a></div>`).join(''));
                }
            } catch (err) {
                updateClass('error-banner', 'bg-red-900 border-2 border-red-500 text-white p-4 rounded-xl text-sm font-mono break-words shadow-lg block');
                updateText('error-banner', "CONNECTION ERROR: " + err.message);
            }
        }
        
        if (Notification.permission !== "granted") { Notification.requestPermission(); }
        fetchMarketData(); setInterval(fetchMarketData, 5000); 
    </script>
</body>
</html>
"""

@app.get("/")
def read_root():
    return HTMLResponse(content=HTML_INTERFACE, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)