from flask import Flask, render_template, request
from flask import redirect
import pandas as pd
import numpy as np
import ccxt

app = Flask(__name__)
exchange = ccxt.binance()

# --- Teknik Göstergeler ---
def get_ohlcv(symbol):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Göstergeler
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_55'] = df['close'].ewm(span=55).mean()
        df['ema_89'] = df['close'].ewm(span=89).mean()
        df['ema_144'] = df['close'].ewm(span=144).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        low_rsi = df['rsi'].rolling(14).min()
        high_rsi = df['rsi'].rolling(14).max()
        stoch_rsi = (df['rsi'] - low_rsi) / (high_rsi - low_rsi)
        df['stoch_k'] = stoch_rsi.rolling(3).mean()
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df['cci'] = (tp - sma) / (0.015 * mad)

        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df['atr'] = atr

        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        df['bb_upper'] = sma + 2 * std
        df['bb_lower'] = sma - 2 * std

        recent_high = df['high'].rolling(20).max()
        recent_low = df['low'].rolling(20).min()
        df['breakout_up'] = df['close'] > recent_high.shift(1)
        df['breakout_down'] = df['close'] < recent_low.shift(1)

        df['supertrend'] = df['close'] > (sma - 2 * std)

        return df
    except:
        return None

@app.route('/')
def home():
    return redirect('/miniapp')

@app.route('/miniapp')
def miniapp():
    return render_template('miniapp.html')


@app.route('/analiz')
def analiz():
    coin = request.args.get("coin", "ETH/USDT").upper()
    market = coin.replace("/", "")
    df = get_ohlcv(market)
    if df is None or df.empty:
        return "Veri alınamadı."

    last = df.iloc[-1]
    entry = last['close']
    stoch_k, stoch_d = last['stoch_k'], last['stoch_d']
    support, resistance = last['support'], last['resistance']

    long_score = short_score = 0
    long_why = []
    short_why = []

    total_score = 15.2
    min_required = 11.5

    if last['supertrend']:
        long_score += 2.6
        long_why.append("✅ Supertrend YUKARI")
    else:
        short_score += 2.6
        short_why.append("✅ Supertrend AŞAĞI")

    if last['ema_8'] > last['ema_21'] > last['ema_55'] > last['ema_89'] > last['ema_144']:
        long_score += 2.4
        long_why.append("✅ EMA Ribbon YUKARI")
    elif last['ema_8'] < last['ema_21'] < last['ema_55'] < last['ema_89'] < last['ema_144']:
        short_score += 2.4
        short_why.append("✅ EMA Ribbon AŞAĞI")

    if last['close'] > last['bb_upper']:
        short_score += 1.0
        short_why.append("✅ BB Üstünde")
    elif last['close'] < last['bb_lower']:
        long_score += 1.0
        long_why.append("✅ BB Altında")
    else:
        long_score += 0.5
        short_score += 0.5

    if last['rsi'] <= 35:
        long_score += 1.0
        long_why.append(f"✅ RSI ({last['rsi']:.2f}) ≤ 35")
    elif last['rsi'] >= 65:
        short_score += 1.0
        short_why.append(f"✅ RSI ({last['rsi']:.2f}) ≥ 65")

    if last['macd'] > last['macd_signal']:
        long_score += 1.5
        long_why.append("✅ MACD > Signal")
    else:
        short_score += 1.5
        short_why.append("✅ MACD < Signal")

    if stoch_k < 0.85 and stoch_d < 0.85:
        long_score += 1.0
        long_why.append("✅ Stoch RSI < 0.85")
    if stoch_k > 0.85 and stoch_d > 0.85:
        short_score += 1.0
        short_why.append("✅ Stoch RSI > 0.85")

    if last['cci'] < -100:
        long_score += 0.5
    elif last['cci'] > 100:
        short_score += 0.5

    if df['obv'].iloc[-1] > df['obv'].iloc[-5]:
        long_score += 0.5
    elif df['obv'].iloc[-1] <= df['obv'].iloc[-5]:
        short_score += 0.5

    if last['breakout_up']:
        long_score += 1.5
    elif last['breakout_down']:
        short_score += 1.5

    if last['close'] > last['ema50'] and last['atr'] > (entry * 0.012):
        long_score += 1.0
    elif last['close'] < last['ema50'] and last['atr'] > (entry * 0.012):
        short_score += 1.0

    if last['atr'] < (entry * 0.012):
        return "📉 ATR düşük, işlem geçersiz"

    if long_score >= min_required:
        pozisyon = "📈 LONG"
    elif short_score >= min_required:
        pozisyon = "📉 SHORT"
    else:
        pozisyon = "📊 NÖTR"

    tp1 = round(entry * 1.03, 4)
    tp2 = round(entry * 1.05, 4)
    tp3 = round(entry * 1.08, 4)
    tp4 = round(entry * 1.12, 4)
    tp5 = round(entry * 1.15, 4)
    sl = round(entry * 0.96, 4)

    yorumlar = long_why if "LONG" in pozisyon else short_why
    yorum = "\n".join(yorumlar)

    return f"""
📊 {coin} için analiz sonucu:

{pozisyon}
Skor: LONG: {long_score:.1f}/{total_score} | SHORT: {short_score:.1f}/{total_score}

Entry: {entry:.4f}  
TP1: {tp1} | TP2: {tp2} | TP3: {tp3} | TP4: {tp4} | TP5: {tp5}  
SL: {sl}  

Detaylar:
{yorum}
"""

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=10000)
