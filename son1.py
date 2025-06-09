import re
import asyncio
from telethon import TelegramClient
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
import requests
import numpy as np
import time
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd


# --- ENV AYARLARI ---
load_dotenv()
api_id = int(os.getenv("API_ID"))
api_hash = os.getenv("API_HASH")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- BALINA (WHALE-ALERT) ENTEGRASYONU ---
WH_ALERT_CHANNEL = 'whale_alert_io'
EXCHANGES = [
    "binance", "kucoin", "okx", "coinbase", "bybit",
    "mexc", "kraken", "bitfinex", "gate.io", "htx", "aave"
]
TIME_FRAMES = [
    ("Son 5 Dakika", 5),
    ("Son 15 Dakika", 15),
    ("Son 30 Dakika", 30),
    ("Son 1 Saat", 60),
    ("Son 4 Saat", 240),
    ("Son 24 Saat", 1440)
]
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "USDT": "tether",
    "PEPE": "pepe",
    "SHIB": "shiba",
    "DOGE": "dogecoin",
    # İstediğin kadar coin ekleyebilirsin
}


def parse_whale_alert(text):
    if not text:
        return None
    m = re.search(
        r'([\d,]+)\s[#]?([A-Za-z0-9]+)[^\n]*\(([\d,]+)\s*USD\).*from (.+?) to (.+?)(?:\.|$|\n)',
        text)
    if not m:
        return None
    amount = float(m.group(1).replace(',', ''))
    coin = m.group(2).upper()
    usd = float(m.group(3).replace(',', ''))
    from_acct = m.group(4).lower()
    to_acct = m.group(5).lower()
    from_is_exchange = any(x in from_acct for x in EXCHANGES)
    to_is_exchange = any(x in to_acct for x in EXCHANGES)
    if to_is_exchange:
        direction = "in"
    elif from_is_exchange:
        direction = "out"
    else:
        direction = "other"
    return {
        "amount": amount,
        "coin": coin,
        "usd": usd,
        "from": from_acct,
        "to": to_acct,
        "direction": direction,
        "from_is_exchange": from_is_exchange,
        "to_is_exchange": to_is_exchange
    }


def analyze_period(messages, t0, t1):
    summary = {}
    xchain_transfers = {}
    for m in messages:
        if not (t0 <= m["date"] < t1):
            continue
        c = m["coin"]
        if c not in summary:
            summary[c] = {
                "in_amount": 0, "out_amount": 0,
                "usd_in": 0, "usd_out": 0,
                "adet_in": 0, "adet_out": 0
            }
        if m["to_is_exchange"] and m["from_is_exchange"]:
            summary[c]["in_amount"] += m["amount"]
            summary[c]["usd_in"] += m["usd"]
            summary[c]["adet_in"] += 1
            summary[c]["out_amount"] += m["amount"]
            summary[c]["usd_out"] += m["usd"]
            summary[c]["adet_out"] += 1
            if c not in xchain_transfers:
                xchain_transfers[c] = []
            xchain_transfers[c].append({
                "amount": m["amount"],
                "usd": m["usd"],
                "from": m["from"],
                "to": m["to"]
            })
        elif m["direction"] == "in":
            summary[c]["in_amount"] += m["amount"]
            summary[c]["usd_in"] += m["usd"]
            summary[c]["adet_in"] += 1
        elif m["direction"] == "out":
            summary[c]["out_amount"] += m["amount"]
            summary[c]["usd_out"] += m["usd"]
            summary[c]["adet_out"] += 1
    return summary, xchain_transfers


def analyze_all_periods(messages, now):
    per_coin = {}
    per_coin_xchain = {}
    for coin in COINGECKO_IDS.keys():
        per_coin[coin] = []
        per_coin_xchain[coin] = []
        for label, minutes in TIME_FRAMES:
            t0 = now - timedelta(minutes=minutes)
            t1 = now
            summary, xchain_transfers = analyze_period(messages, t0, t1)
            data = summary.get(coin, {
                "in_amount": 0, "out_amount": 0,
                "usd_in": 0, "usd_out": 0,
                "adet_in": 0, "adet_out": 0
            })
            per_coin[coin].append((label, data))
            if coin in xchain_transfers:
                per_coin_xchain[coin].append((label, xchain_transfers[coin]))
            else:
                per_coin_xchain[coin].append((label, []))
    return per_coin, per_coin_xchain

# --- API RETRY --- #


def safe_api_call(func, max_retry=5, wait=5, *args, **kwargs):
    last_error = None
    for _ in range(max_retry):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            last_error = e
            time.sleep(wait)
    return None if last_error is None else (None, str(last_error))


def get_daily_volume_usd(coin):
    if coin not in COINGECKO_IDS:
        return None, "ID yok"
    cg_id = COINGECKO_IDS[coin]
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}?localization=false&tickers=false&market_data=true"
    for _ in range(5):
        try:
            r = requests.get(url, timeout=15)
            data = r.json()
            if "status" in data and data["status"].get("error_code") == 429:
                time.sleep(10)
                continue
            time.sleep(3)
            market_data = data.get("market_data")
            if not market_data:
                return None, "market_data yok"
            total_volume = market_data.get("total_volume")
            if not total_volume or "usd" not in total_volume:
                return None, "total_volume yok"
            return float(total_volume["usd"]), None
        except Exception as e:
            time.sleep(5)
    return None, "get_daily_volume_usd hata"


def get_daily_price(coin):
    if coin not in COINGECKO_IDS:
        return None, "ID yok"
    cg_id = COINGECKO_IDS[coin]
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}?localization=false&tickers=false&market_data=true"
    for _ in range(5):
        try:
            r = requests.get(url, timeout=15)
            data = r.json()
            if "status" in data and data["status"].get("error_code") == 429:
                time.sleep(10)
                continue
            time.sleep(3)
            market_data = data.get("market_data")
            if not market_data:
                return None, "market_data yok"
            current_price = market_data.get("current_price")
            if not current_price or "usd" not in current_price:
                return None, "current_price yok"
            return float(current_price["usd"]), None
        except Exception as e:
            time.sleep(5)
    return None, "get_daily_price hata"


def yorum_uret(fark, gunluk_hacim, yon, hacim_var):
    if not hacim_var:
        return "⚠️ CoinGecko veri eksik, oran ve öneri üretilemedi."
    oran = 0
    if gunluk_hacim and gunluk_hacim > 0:
        oran = abs(fark) / gunluk_hacim * 100
    oran_s = f"{oran:.2f}%"
    if fark == 0 or gunluk_hacim is None or gunluk_hacim == 0:
        return f"🟡 Baskı yok, piyasa nötr. (Günlük hacme oran: {oran_s})"
    if yon == 'out':
        if oran < 1:
            return f"🟡 Hafif alım baskısı var, piyasa yatay veya nötr. (Günlük hacme oran: {oran_s})"
        elif oran < 5:
            return f"🟢 Alım baskısı hissediliyor, hareket başlayabilir. (Günlük hacme oran: {oran_s})"
        else:
            return f"🟢 Güçlü alım baskısı! Piyasa alıma dönüyor, hareketli gün olabilir. (Günlük hacme oran: {oran_s})"
    else:
        if oran < 1:
            return f"🟡 Hafif satış baskısı var, piyasa yatay veya nötr. (Günlük hacme oran: {oran_s})"
        elif oran < 5:
            return f"🔴 Satış baskısı hissediliyor, hareket başlayabilir. (Günlük hacme oran: {oran_s})"
        else:
            return f"🔴 Güçlü satış baskısı! Piyasa satıma dönüyor, dikkatli ol. (Günlük hacme oran: {oran_s})"


def get_period_yon(data):
    fark = data['in_amount'] - data['out_amount']
    if fark > 0:
        return 'in'
    elif fark < 0:
        return 'out'
    else:
        return None


def format_btc_whale_report(all_period_data, all_xchain_data,
                            gunluk_hacim, gunluk_fiyat, hacim_var, hacim_error, now_tr):
    out = [f"\n━━ 🐋 Balina Transfer Analizi ━━"]
    out.append(f"Tarih/Saat (TSI): {now_tr}\n")
    if not hacim_var:
        out.append(
            f"⚠️ CoinGecko veri eksik: {hacim_error}. Bu coin için oran ve öneri gösterilemiyor.")
    son_data = all_period_data[-1][1]
    fark = son_data["in_amount"] - son_data["out_amount"]
    yon = get_period_yon(son_data)
    genel_yorum = yorum_uret(
        fark *
        gunluk_fiyat if gunluk_fiyat else fark,
        gunluk_hacim,
        yon,
        hacim_var)
    out.append(genel_yorum)
    for i, (label, data) in enumerate(all_period_data):
        fark_amount = data["in_amount"] - data["out_amount"]
        fark_usd = data["usd_in"] - data["usd_out"]
        yon = get_period_yon(data)
        oran = (
            abs(fark_usd) /
            gunluk_hacim *
            100) if (
            gunluk_hacim and hacim_var) else 0
        oran_s = f"{oran:.2f}%"
        out.append(
            f"\n{label}\n"
            f"  • Borsaya giriş: {data['in_amount']:,.2f} BTC ({data['adet_in']} işlem), {data['usd_in']:,.0f} USD\n"
            f"  • Borsadan çıkış: {data['out_amount']:,.2f} BTC ({data['adet_out']} işlem), {data['usd_out']:,.0f} USD\n"
            f"  • Fark: {fark_amount:,.2f} BTC, {fark_usd:,.0f} USD (Günlük hacme oran: {oran_s if hacim_var else 'Veri yok'})"
        )
        xchain_label, xchain_list = all_xchain_data[i]
        for x in xchain_list:
            out.append(
                f"    ↪️ Ekstra: {x['amount']:,.2f} BTC ({x['usd']:,.0f} USD) {label} diliminde {x['from']} platformundan {x['to']} platformuna transfer edildi."
            )
        yorum = yorum_uret(fark_usd, gunluk_hacim, yon, hacim_var)
        out.append("    " + yorum)
    return "\n".join(out)


def format_all_coins_whale_report(
        per_coin, per_coin_xchain, gunluk_hacimler, gunluk_fiyatlar, now_tr):
    out = ["\n━━ 🐋 Balina Transfer Analizi (Tüm Coinler) ━━"]
    out.append(f"Tarih/Saat (TSI): {now_tr}\n")
    for coin in per_coin:
        if coin not in COINGECKO_IDS:
            continue
        if coin == "BTC":
            continue  # BTC zaten yukarıda detaylı veriliyor
        all_period_data = per_coin[coin]
        gunluk_hacim = gunluk_hacimler.get(coin)
        gunluk_fiyat = gunluk_fiyatlar.get(coin)
        hacim_var = gunluk_hacim is not None
        son_data = all_period_data[-1][1]
        fark = son_data["in_amount"] - son_data["out_amount"]
        yon = get_period_yon(son_data)
        genel_yorum = yorum_uret(
            fark *
            gunluk_fiyat if gunluk_fiyat else fark,
            gunluk_hacim,
            yon,
            hacim_var)
        out.append(f"\n[{coin}] {genel_yorum}")
        for i, (label, data) in enumerate(all_period_data):
            fark_amount = data["in_amount"] - data["out_amount"]
            fark_usd = data["usd_in"] - data["usd_out"]
            yon = get_period_yon(data)
            oran = (
                abs(fark_usd) /
                gunluk_hacim *
                100) if (
                gunluk_hacim and hacim_var) else 0
            oran_s = f"{oran:.2f}%"
            out.append(
                f"{label}: Giriş: {data['in_amount']:,.2f} {coin} | Çıkış: {data['out_amount']:,.2f} {coin} | Fark: {fark_amount:,.2f} {coin} | Oran: {oran_s if hacim_var else '-'}"
            )
        out.append("-" * 40)
    return "\n".join(out)


def get_order_book_depth(symbol="BTCUSDT", limit=20):
    try:
        url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
        data = requests.get(url, timeout=10).json()
        bids = sum(float(x[1]) for x in data["bids"])
        asks = sum(float(x[1]) for x in data["asks"])
        return bids, asks
    except Exception:
        return None, None


def send_telegram_message_split(msg, max_len=4000, header=""):
    """
    Telegram mesajını paragraflara veya bölücü çizgilerine (örn. '─────', '━━') göre bölerek gönderir.
    Bölümler arasına otomatik boşluk ekler, başlık ekler, bölümler karışmaz.
    """
    # Bölücü anahtar kelimeleri
    dividers = ["─────", "━━", "Backtest Sonucu", "Nihai Öneri", "Balina Transfer Analizi"]

    # Her bölücünün önüne ve sonuna boş satır ekle
    for d in dividers:
        msg = msg.replace(d, f"\n{d}\n")

    # Paragraflara böl
    paragraphs = []
    buf = ""
    for line in msg.splitlines(keepends=True):
        buf += line
        if line.strip() == "" or any(div in line for div in dividers):
            if buf.strip():
                paragraphs.append(buf)
            buf = ""
    if buf.strip():
        paragraphs.append(buf)

    # Paragrafları max_len karakteri aşmayacak şekilde birleştir
    parts = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) > max_len:
            parts.append(current)
            current = ""
        current += p
    if current:
        parts.append(current)

    total = len(parts)
    for idx, part in enumerate(parts, 1):
        head = f"{header} ({idx}/{total})\n\n" if header else f"({idx}/{total})\n\n"
        text = head + part.strip()
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text
        }
        try:
            r = requests.post(url, data=data, timeout=10)
            if r.status_code != 200:
                print(f"Telegram mesajı gönderilemedi: {r.text}")
            else:
                print("Telegram'a mesaj gönderildi.")
        except Exception as e:
            print(f"Telegram gönderim hatası: {e}")

# --- TEKNİK ANALİZ GÖSTERGELERİ (ATR VE GELİŞMİŞ FİLTRELER ENTEGRE) ---
np.seterr(divide='ignore', invalid='ignore')


def ema(arr, n):
    arr = np.array(arr)
    if len(arr) < n:
        return None
    ema_arr = np.zeros_like(arr)
    ema_arr[0] = arr[0]
    alpha = 2 / (n + 1)
    for i in range(1, len(arr)):
        ema_arr[i] = alpha * arr[i] + (1 - alpha) * ema_arr[i - 1]
    return ema_arr


def macd(arr, fast=12, slow=26, signal=9):
    if len(arr) < max(fast, slow, signal):
        return None, None, None
    macd_line = ema(arr, fast) - ema(arr, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(arr, period=14):
    arr = np.array(arr)
    if len(arr) < period + 1:
        return None
    deltas = np.diff(arr)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi_arr = np.zeros_like(arr)
    rsi_arr[:period] = 50
    for i in range(period, len(arr)):
        delta = deltas[i - 1]
        upval = max(delta, 0)
        downval = -min(delta, 0)
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi_arr[i] = 100 - 100 / (1 + rs) if down != 0 else 100
    return rsi_arr


def stoch_rsi(arr, period=14):
    arr_rsi = rsi(arr, period)
    if arr_rsi is None:
        return None
    stoch = np.zeros_like(arr_rsi)
    for i in range(period, len(arr_rsi)):
        lowest = np.min(arr_rsi[i - period + 1:i + 1])
        highest = np.max(arr_rsi[i - period + 1:i + 1])
        stoch[i] = 100 * (arr_rsi[i] - lowest) / (highest -
                                                  lowest) if highest != lowest else 0
    return stoch


def mfi(high, low, close, volume, period=14):
    high, low, close, volume = map(np.array, (high, low, close, volume))
    if len(close) < period + 1:
        return None
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    mfi_arr = np.zeros_like(close)
    for i in range(period, len(close)):
        pos_mf = 0
        neg_mf = 0
        for j in range(i - period + 1, i + 1):
            if tp[j] > tp[j - 1]:
                pos_mf += raw_mf[j]
            elif tp[j] < tp[j - 1]:
                neg_mf += raw_mf[j]
        mfr = pos_mf / neg_mf if neg_mf != 0 else 0
        mfi_arr[i] = 100 - 100 / (1 + mfr) if neg_mf != 0 else 100
    return mfi_arr


def adx(high, low, close, period=14):
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    if len(close) < period + 1:
        return None
    plus_dm = np.zeros_like(close)
    minus_dm = np.zeros_like(close)
    tr = np.zeros_like(close)
    for i in range(1, len(close)):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    tr_ema = ema(tr, period)
    plus_di = np.nan_to_num(100 * ema(plus_dm, period) / tr_ema)
    minus_di = np.nan_to_num(100 * ema(minus_dm, period) / tr_ema)
    dx = np.nan_to_num(100 * np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx_arr = ema(dx, period)
    return adx_arr


def obv(close, volume):
    close = np.array(close)
    volume = np.array(volume)
    if len(close) < 2:
        return None
    obv_arr = np.zeros_like(close)
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv_arr[i] = obv_arr[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv_arr[i] = obv_arr[i - 1] - volume[i]
        else:
            obv_arr[i] = obv_arr[i - 1]
    return obv_arr


def bollinger(arr, period=20, dev=2):
    arr = np.array(arr)
    if len(arr) < period:
        return None, None, None
    ma = np.zeros_like(arr)
    upper = np.zeros_like(arr)
    lower = np.zeros_like(arr)
    for i in range(period - 1, len(arr)):
        ma[i] = np.mean(arr[i - period + 1:i + 1])
        std = np.std(arr[i - period + 1:i + 1])
        upper[i] = ma[i] + dev * std
        lower[i] = ma[i] - dev * std
    return ma, upper, lower


def atr(high, low, close, period=14):
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    if len(close) < period + 1:
        return None
    tr = np.zeros_like(close)
    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    atr_arr = ema(tr, period)
    return atr_arr


def volatility_level(atr_now, close_now, vade_label=None):
    if atr_now is None or close_now is None or close_now == 0:
        return "Veri yok"
    ratio = atr_now / close_now
    if vade_label and "5dk" in vade_label:
        if ratio < 0.01:
            return f"Çok Düşük (5dk ATR: {atr_now:.2f})"
        elif ratio < 0.02:
            return f"Düşük (5dk ATR: {atr_now:.2f})"
        else:
            return f"Yüksek (5dk ATR: {atr_now:.2f})"
    if ratio < 0.01:
        return f"Düşük (ATR: {atr_now:.2f})"
    elif ratio < 0.02:
        return f"Orta (ATR: {atr_now:.2f})"
    else:
        return f"Yüksek (ATR: {atr_now:.2f})"


def trend_strength_text(trend, adx_val):
    if adx_val is None:
        return f"{trend} (ADX veri yok)"
    if adx_val < 20:
        return f"{trend}, zayıf trend (ADX {adx_val:.2f})"
    elif adx_val < 25:
        return f"{trend}, orta trend (ADX {adx_val:.2f})"
    else:
        return f"{trend}, güçlü trend (ADX {adx_val:.2f})"


def btc_kisavadeli_analizler(ohlcv_dict, current_price, dtstr_tr, dtstr_utc):
    results = []
    vadeler = [
        ("5dk", ohlcv_dict.get("5m")),
        ("15dk", ohlcv_dict.get("15m")),
        ("30dk", ohlcv_dict.get("30m"))
    ]
    for vade, ohlcv in vadeler:
        if not ohlcv or len(ohlcv["close"]) < 25:
            results.append(f"📉 {vade} Analiz: Veri yok")
            continue
        close = np.array(ohlcv['close'])
        high = np.array(ohlcv['high'])
        low = np.array(ohlcv['low'])
        volume = np.array(ohlcv['volume'])
        ema7 = ema(close, 7)[-1] if ema(close, 7) is not None else None
        ema21 = ema(close, 21)[-1] if ema(close, 21) is not None else None
        macd_line = macd(close, 12, 26, 9)[
            0][-1] if macd(close, 12, 26, 9)[0] is not None else None
        rsi_val = rsi(close, 14)[-1] if rsi(close, 14) is not None else None
        mfi_val = mfi(high,
                      low,
                      close,
                      volume,
                      14)[-1] if mfi(high,
                                     low,
                                     close,
                                     volume,
                                     14) is not None else None
        atr_val = atr(high,
                      low,
                      close,
                      14)[-1] if atr(high,
                                     low,
                                     close,
                                     14) is not None else None
        vol_txt = volatility_level(
            atr_val, close[-1] if len(close) else None, vade_label=vade)

        trend = None
        if ema7 is not None and ema21 is not None:
            if ema7 > ema21:
                trend = "Pozitif"
            elif ema7 < ema21:
                trend = "Negatif"
        score = 0
        max_score = 0
        if ema7 is not None and ema21 is not None:
            max_score += 2
            if ema7 > ema21:
                score += 2
            else:
                score -= 2
        if macd_line is not None:
            max_score += 2
            if macd_line > 0:
                score += 2
            else:
                score -= 2
        if rsi_val is not None:
            max_score += 1
            if rsi_val > 50:
                score += 1
            else:
                score -= 1
        if mfi_val is not None:
            max_score += 1
            if mfi_val > 50:
                score += 1
            else:
                score -= 1
        if atr_val is not None:
            max_score += 1
            if vol_txt.startswith("Yüksek") or vol_txt.startswith("Çok Düşük"):
                score -= 1
            elif vol_txt.startswith("Düşük"):
                score += 1
        if max_score == 0:
            signal = "⚪️"
            strength = "Veri yok"
        else:
            ratio = score / max_score
            if ratio >= 0.7:
                signal = "🟢"
                strength = "GÜÇLÜ"
            elif ratio >= 0.3:
                signal = "🟡"
                strength = "ORTA"
            elif ratio >= 0:
                signal = "🟡"
                strength = "ZAYIF"
            else:
                signal = "🔴"
                strength = "SAT"
        results.append(
            f"📉 {vade} Analiz: {signal} {strength} | EMA7/21: {'Pozitif' if trend == 'Pozitif' else 'Negatif' if trend == 'Negatif' else 'Veri yok'} | MACD: {'Pozitif' if macd_line is not None and macd_line > 0 else 'Negatif' if macd_line is not None and macd_line < 0 else 'Veri yok'} | RSI: {rsi_val:.2f} | ATR: {atr_val:.2f} | Volatilite: {vol_txt}"
        )
        if vade == "5dk":
            results.append(
                "⚠️ 5dk'lık analizlerde volatilite ve ATR genellikle düşüktür, ani hareketler yanıltıcı olabilir.")
    return "━━ Kısa Vadeli BTC Analizleri ━━\n" + "\n".join(results) + "\n\n"


def btc_teknik_analiz_raporu(
    ohlcv,
    current_price,
    dtstr_tr,
    dtstr_utc,
    balina_net_1h,
    ls_ratio_1h,
    vade="1 Saatlik Analiz"
):
    close = np.array(ohlcv['close'])
    high = np.array(ohlcv['high'])
    low = np.array(ohlcv['low'])
    volume = np.array(ohlcv['volume'])

    ema7_arr = ema(close, 7)
    ema21_arr = ema(close, 21)
    ema7 = ema7_arr[-1] if isinstance(ema7_arr, np.ndarray) else None
    ema21 = ema21_arr[-1] if isinstance(ema21_arr, np.ndarray) else None

    macd_out = macd(close, 12, 26, 9)
    macd_line = macd_out[0][-1] if isinstance(
        macd_out[0], np.ndarray) else None

    rsi_arr = rsi(close, 14)
    rsi_val = rsi_arr[-1] if isinstance(rsi_arr, np.ndarray) else None

    stochrsi_arr = stoch_rsi(close, 14)
    stochrsi_val = stochrsi_arr[-1] if isinstance(
        stochrsi_arr, np.ndarray) else None

    mfi_arr = mfi(high, low, close, volume, 14)
    mfi_val = mfi_arr[-1] if isinstance(mfi_arr, np.ndarray) else None

    adx_arr = adx(high, low, close, 14)
    adx_val = adx_arr[-1] if isinstance(adx_arr, np.ndarray) else None

    obv_arr = obv(close, volume)
    obv_val = obv_arr[-1] if isinstance(obv_arr, np.ndarray) else None
    obv_prev = obv_arr[-2] if (isinstance(obv_arr,
                                          np.ndarray) and len(obv_arr) > 1) else None
    obv_1h_pct = (100 * (obv_val - obv_prev) / abs(obv_prev)) if (
        obv_val is not None and obv_prev is not None and abs(obv_prev) > 0) else None

    boll_ma, boll_up, boll_down = bollinger(close, 20, 2)
    atr_arr = atr(high, low, close, 14)
    atr_now = atr_arr[-1] if isinstance(atr_arr, np.ndarray) else None
    volatility_txt = volatility_level(
        atr_now, close[-1] if len(close) else None)

    trend = None
    if ema7 is not None and ema21 is not None:
        if ema7 > ema21:
            trend = "YUKARI"
        elif ema7 < ema21:
            trend = "AŞAĞI"
    trend_guc_txt = trend_strength_text(trend or "N/A", adx_val)
    trend_guc_score = None
    if adx_val is not None:
        if adx_val < 20:
            trend_guc_score = 0
        elif adx_val < 25:
            trend_guc_score = 1
        else:
            trend_guc_score = 2

    score = 0
    max_score = 0
    missing = []
    if ema7 is not None and ema21 is not None:
        max_score += 2
        if ema7 > ema21:
            score += 2
        else:
            score -= 2
    else:
        missing.append("EMA")

    if macd_line is not None:
        max_score += 2
        if macd_line > 0:
            score += 2
        else:
            score -= 2
    else:
        missing.append("MACD")

    if rsi_val is not None:
        max_score += 1
        if rsi_val > 50:
            score += 1
        else:
            score -= 1
    else:
        missing.append("RSI")

    if mfi_val is not None:
        max_score += 1
        if mfi_val > 50:
            score += 1
        else:
            score -= 1
    else:
        missing.append("MFI")

    if adx_val is not None:
        max_score += 1
        if adx_val > 25:
            score += 1
    else:
        missing.append("ADX")

    if obv_val is not None:
        max_score += 2  # OBV etkisi artırıldı
        if obv_val > 0:
            score += 2
        else:
            score -= 2
    else:
        missing.append("OBV")

    if atr_now is not None:
        max_score += 1
        if volatility_txt.startswith("Yüksek"):
            score -= 1
        elif volatility_txt.startswith("Düşük"):
            score += 1
    else:
        missing.append("ATR")

    if trend_guc_score is not None:
        max_score += 1
        score += trend_guc_score - 1

    max_score += 1
    if balina_net_1h > 0:
        score -= 1
    elif balina_net_1h < 0:
        score += 1

    max_score += 1
    if ls_ratio_1h > 1.10:
        score += 1
    elif ls_ratio_1h < 0.90:
        score -= 1

    if max_score == 0:
        signal_strength = "Veri Yok"
        signal = "⚪️ TUT (Veri Yok)"
    else:
        ratio = score / max_score
        if ratio >= 0.85:
            signal_strength = "ÇOK GÜÇLÜ"
        elif ratio >= 0.6:
            signal_strength = "ORTA"
        elif ratio >= 0.3:
            signal_strength = "ZAYIF"
        else:
            signal_strength = "TEREDDÜTLÜ"
        if score <= -3:
            signal = f"🔴 SAT ({signal_strength})"
        elif score >= 5:
            signal = f"🟢 AL ({signal_strength})"
        elif score >= 1:
            signal = f"🟡 TUT/AL ({signal_strength})"
        elif score <= -1:
            signal = f"🟡 TUT/SAT ({signal_strength})"
        else:
            signal = f"⚪️ TUT ({signal_strength})"

    destek = min(close[-20:]) if len(close) >= 20 else min(close)
    direnç = max(close[-20:]) if len(close) >= 20 else max(close)

    balina_etiket = "Pozitif (Borsadan çıkış)" if balina_net_1h < 0 else "Negatif (Borsaya giriş)"
    ls_etiket = "Pozitif (Longlar baskın)" if ls_ratio_1h > 1.05 else "Negatif (Shortlar baskın)"

    ek_veriler = (
        f"\n📊 Ek Veriler ({vade})\n"
        f"• Balina Net Akışı: {balina_net_1h:.2f} BTC ({balina_etiket})\n"
        f"• Long/Short Oranı: {ls_ratio_1h:.2f} ({ls_etiket})\n"
        f"• OBV değişim (1h): {obv_1h_pct:+.2f}%" if obv_1h_pct is not None else "• OBV değişim (1h): Veri yok"
        + f"\n• {trend_guc_txt}"
        + f"\n• Volatilite: {volatility_txt}"
    )

    rapor = []
    rapor.append(f"💹 BTC Teknik Analiz ({dtstr_tr})")
    rapor.append(f"Fiyat: ${current_price:,.2f}")
    rapor.append(f"Zaman: {dtstr_tr} (TR) / {dtstr_utc} UTC")
    rapor.append("─────")
    rapor.append(f"📊 {vade}")
    rapor.append(f"Sinyal: {signal} (Skor: {score}/{max_score})")
    rapor.append(f"• EMA7: {ema7:.2f} | EMA21: {ema21:.2f} → {'Negatif' if ema7 is not None and ema21 is not None and ema7 < ema21 else 'Pozitif' if ema7 is not None and ema21 is not None else 'Veri yok'}")
    rapor.append(f"• MACD: {macd_line:.2f} → {'Negatif, momentum aşağı.' if macd_line is not None and macd_line < 0 else 'Pozitif, momentum yukarı.' if macd_line is not None and macd_line >= 0 else 'Veri yok'}")
    rapor.append(
        f"• RSI: {rsi_val:.2f}" if rsi_val is not None else "• RSI: Veri yok")
    rapor.append(
        f"• StochRSI: {stochrsi_val:.2f}" if stochrsi_val is not None else "• StochRSI: Veri yok")
    rapor.append(
        f"• MFI: {mfi_val:.2f}" if mfi_val is not None else "• MFI: Veri yok")
    rapor.append(
        f"• ADX: {adx_val:.2f} → {'Güçlü trend var.' if adx_val is not None and adx_val > 25 else 'Trend zayıf.' if adx_val is not None else 'Veri yok'}")
    rapor.append(
        f"• OBV: {obv_val:.2f} → {'Alış baskısı var.' if obv_val is not None and obv_val > 0 else 'Satış baskısı var.' if obv_val is not None and obv_val < 0 else 'Veri yok'}")
    if boll_ma is not None and boll_up is not None and boll_down is not None:
        rapor.append(
            f"• Bollinger: MA {boll_ma[-1]:.2f} | Üst {boll_up[-1]:.2f} | Alt {boll_down[-1]:.2f}")
    else:
        rapor.append("• Bollinger: Veri yok")
    rapor.append(f"• Trend filtresi: {trend_guc_txt}")
    rapor.append(f"• Volatilite: {volatility_txt}")
    rapor.append(f"• Destek: ${destek:,.2f} | Direnç: ${direnç:,.2f}")
    rapor.append(ek_veriler)
    if missing:
        rapor.append(
            f"\n⚠️ Eksik gösterge(ler): {', '.join(missing)} (Bu göstergeler skora katılmadı)")
    return "\n".join(
        rapor), score, max_score, destek, direnç, ema7, ema21, macd_line, rsi_val, obv_val, trend, obv_1h_pct

# --- PİYASA VERİLERİ (FUNDING, RATIO, OI, HACİM, ORDERBOOK) ---


def get_funding_rate(symbol="BTCUSDT"):
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
    try:
        result = requests.get(url, timeout=10).json()
        return float(result[0]['fundingRate'])
    except Exception:
        return None


def get_open_interest(symbol="BTCUSDT"):
    url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
    try:
        result = requests.get(url, timeout=10).json()
        return float(result['openInterest'])
    except Exception:
        return None


def get_long_short_ratio(symbol="BTCUSDT", period="5m"):
    url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period={period}&limit=1"
    try:
        result = requests.get(url, timeout=10).json()
        ratio = float(result[0]['longShortRatio'])
        return ratio
    except Exception:
        return None


def get_spot_volume(symbol="BTCUSDT", interval="5m", count=1):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={count}"
    try:
        data = requests.get(url, timeout=10).json()
        total = sum(float(x[5]) for x in data)
        return total
    except Exception:
        return None


def get_futures_volume(symbol="BTCUSDT", interval="5m", count=1):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={count}"
    try:
        data = requests.get(url, timeout=10).json()
        total = sum(float(x[7]) for x in data)
        return total
    except Exception:
        return None


def btc_piyasa_analiz_turkce():
    intervals = [
        ("5m", "Son 5 Dakika"),
        ("30m", "Son 30 Dakika"),
        ("1h", "Son 1 Saat"),
        ("4h", "Son 4 Saat"),
        ("1d", "Son 24 Saat")
    ]
    out = "━━ BTC Piyasa Verileri ━━\n"
    for interval, label in intervals:
        funding_rate = get_funding_rate()
        ratio = get_long_short_ratio(period=interval)
        open_interest = get_open_interest()
        spot_vol = get_spot_volume(interval=interval, count=1)
        futures_vol = get_futures_volume(interval=interval, count=1)
        bids, asks = get_order_book_depth(limit=20)
        any_data = False
        lines = [f"\n📅 {label}"]
        if funding_rate is not None:
            lines.append(
                f"• Fonlama Oranı: {funding_rate:.5f} ({'Pozitif' if funding_rate > 0 else 'Negatif'})\n  (Vadeli işlem fonlama oranı. Negatif ise short pozisyonlar daha baskın.)")
            any_data = True
        if ratio is not None:
            lines.append(
                f"• Uzun/Kısa Oranı: {ratio:.2f} (1'in altı short ağırlık demektir.)")
            any_data = True
        if open_interest is not None:
            lines.append(
                f"• Açık Pozisyon: {open_interest:,.0f} BTC (Piyasadaki toplam açık kontrat miktarı.)")
            any_data = True
        if spot_vol is not None:
            lines.append(f"• Spot İşlem Hacmi: {spot_vol:,.2f} BTC")
            any_data = True
        if futures_vol is not None:
            lines.append(f"• Vadeli İşlem Hacmi: {futures_vol:,.2f} USD")
            any_data = True
        if bids is not None and asks is not None:
            lines.append(
                f"• Emir Derinliği: Alış: {bids:.2f} BTC | Satış: {asks:.2f} BTC")
            any_data = True
        if any_data:
            out += "\n".join(lines) + "\n"
    return out


def nihai_oneri(
    skor_1h, skor_4h, skor_1d,
    balina_net_1h, ls_ratio_1h,
    obv_delta_1h, price_delta_1h
):
    """
    Dinamik nihai öneri fonksiyonu: teknik analiz, balina akışı, long/short oranı, OBV ve fiyat değişimini dikkate alır.
    """
    karar = "TUT"
    simge = "🟡"
    gerekce = []

    # Güçlü AL koşulu
    if (
        skor_1h >= 4 and skor_4h >= 2 and skor_1d >= 2 and
        balina_net_1h < 0 and obv_delta_1h > 0 and price_delta_1h > 0
    ):
        karar = "AL"
        simge = "🟢"
        gerekce.append("Tüm vadelerde göstergeler pozitif.")
        gerekce.append("Balinalar borsadan çıkıyor (pozitif akış).")
        gerekce.append("OBV artıyor (güçlü alım baskısı).")
        gerekce.append("Fiyat artıyor.")
        if ls_ratio_1h < 0.7:
            gerekce.append("Shortlar baskın, short squeeze ihtimali yüksek.")

    # Güçlü SAT koşulu
    elif (
        skor_1h <= -2 and skor_4h <= -1 and skor_1d < 0 and
        balina_net_1h > 0 and obv_delta_1h < 0 and price_delta_1h < 0
    ):
        karar = "SAT"
        simge = "🔴"
        gerekce.append("Çoklu vadede göstergeler negatif.")
        gerekce.append("Balinalar borsaya giriyor (negatif akış).")
        gerekce.append("OBV düşüyor (satış baskısı).")
        gerekce.append("Fiyat düşüyor.")
        if ls_ratio_1h > 1.3:
            gerekce.append("Longlar baskın, aşağı yönlü kırılım riski var.")

    # Teknik analiz pozitif ama balina girişi veya OBV zayıf
    elif (skor_1h >= 3 and (balina_net_1h > 0 or obv_delta_1h < 0)):
        karar = "DİKKATLİ AL"
        simge = "🟡"
        gerekce.append("Göstergeler pozitif ama balinalar borsaya giriyor veya OBV zayıf. Kısa vadede risk olabilir.")

    # Teknik analiz negatif ama balina çıkışı veya OBV güçlü
    elif (skor_1h <= -1 and (balina_net_1h < 0 or obv_delta_1h > 0)):
        karar = "DİKKATLİ SAT"
        simge = "🟡"
        gerekce.append("Göstergeler negatif ama balinalar borsadan çıkıyor veya OBV güçlü. Kısa vadede yukarı sıçrama riski var.")

    # Yatay/kararsız bölge
    elif abs(price_delta_1h) < 0.5 and abs(obv_delta_1h) < 3 and abs(balina_net_1h) < 200:
        karar = "TUT"
        simge = "🟡"
        gerekce.append("Fiyat ve hacim hareketleri zayıf, piyasa kararsız/yatay.")

    # Default
    else:
        karar = "TUT"
        simge = "🟡"
        gerekce.append("Kısa ve uzun vade arasında kararsızlık var, acele etme.")

    return f"\n📢 Nihai Öneri: {simge} {karar}\nGerekçe: {' '.join(gerekce)}\n"


def get_spot_ohlcv(symbol="BTCUSDT", interval="1h", limit=200):
    import requests
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    r = requests.get(url, timeout=10)
    data = r.json()
    ohlcv = {
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
        "ts": []
    }
    for kline in data:
        try:
            ohlcv["open"].append(float(kline[1]))
            ohlcv["high"].append(float(kline[2]))
            ohlcv["low"].append(float(kline[3]))
            ohlcv["close"].append(float(kline[4]))
            ohlcv["volume"].append(float(kline[5]))
            ohlcv["ts"].append(int(kline[0]))
        except (ValueError, IndexError, TypeError):
            continue  # Bozuk/hatalı satırı atla
    return ohlcv

# --------------------------
# YENİ EKLENEN FONKSİYONLAR
# --------------------------


def generate_dynamic_comment(rsi_val, macd_val, obv_pct):
    """Dinamik yorum üretir (RSI, MACD, OBV'ye göre)"""
    comments = []
    if rsi_val > 70:
        comments.append("⚠️ RSI aşırı alım bölgesinde")
    elif rsi_val < 30:
        comments.append("🟢 RSI aşırı satım bölgesinde")

    if macd_val > 0:
        comments.append("MACD yukarı momentumu gösteriyor")
    else:
        comments.append("MACD aşağı momentumu gösteriyor")

    if obv_pct and obv_pct > 5:
        comments.append(f"OBV %{obv_pct:.1f} arttı (güçlü alım)")
    return ". ".join(comments) + "." if comments else "Belirgin sinyal yok."


def plot_technical_indicators(ohlcv):
    """Fiyat ve göstergeleri grafikle gösterir"""
    plt.figure(figsize=(12, 8))
    closes = ohlcv['close'][-100:]  # Son 100 mum

    # 1) Fiyat ve EMA'lar
    plt.subplot(3, 1, 1)
    plt.plot(closes, label='Fiyat', color='blue')
    plt.plot(ema(closes, 7)[-100:], label='EMA7', linestyle='--')
    plt.plot(ema(closes, 21)[-100:], label='EMA21', linestyle='--')
    plt.title('Fiyat ve EMA Trendi')
    plt.legend()

    # 2) RSI ve MACD
    plt.subplot(3, 1, 2)
    plt.plot(rsi(closes, 14)[-100:], label='RSI', color='purple')
    plt.axhline(70, linestyle='--', color='red')
    plt.axhline(30, linestyle='--', color='green')
    plt.title('RSI Göstergesi')

    # 3) Hacim
    plt.subplot(3, 1, 3)
    plt.bar(range(len(ohlcv['volume'][-100:])),
            ohlcv['volume'][-100:], color='gray')
    plt.title('İşlem Hacmi')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


def backtest_strategy(ohlcv_data, lookback=50, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, take_profit=0.01, stop_loss=0.01, hold_bars=5):
    closes = ohlcv_data['close']
    total_signals = 0
    total_success = 0
    for i in range(lookback, len(closes) - hold_bars):
        current_slice = {
            'close': closes[i - lookback:i],
            'high': ohlcv_data['high'][i - lookback:i],
            'low': ohlcv_data['low'][i - lookback:i],
            'volume': ohlcv_data['volume'][i - lookback:i]
        }
        rsi_val = rsi(current_slice['close'], rsi_period)[-1]
        macd_line = macd(current_slice['close'], macd_fast, macd_slow, macd_signal)[0][-1]
        if rsi_val < 30 and macd_line > 0:
            entry = closes[i]
            # TP/SL backtest
            target = entry * (1 + take_profit)
            stop = entry * (1 - stop_loss)
            for j in range(1, hold_bars + 1):
                bar_high = ohlcv_data['high'][i + j]
                bar_low = ohlcv_data['low'][i + j]
                if bar_high >= target:
                    total_success += 1
                    break
                elif bar_low <= stop:
                    break
            total_signals += 1
    success_rate = (total_success / total_signals * 100) if total_signals > 0 else 0
    return success_rate, total_signals


async def main_loop():
    client = TelegramClient('whalealert_session', api_id, api_hash)
    await client.start()
    all_messages = []
    async for msg in client.iter_messages(WH_ALERT_CHANNEL, limit=1000):
        if not msg.message:
            continue
        parsed = parse_whale_alert(msg.message)
        if parsed and parsed["coin"] in COINGECKO_IDS:
            parsed["date"] = msg.date
            all_messages.append(parsed)

    while True:
        if all_messages:
            last_date = max(m["date"] for m in all_messages)
        else:
            last_date = datetime.utcnow() - timedelta(days=10)

        new_messages = []
        async for msg in client.iter_messages(WH_ALERT_CHANNEL, limit=100):
            if not msg.message:
                continue
            if msg.date <= last_date:
                break
            parsed = parse_whale_alert(msg.message)
            if parsed and parsed["coin"] in COINGECKO_IDS:
                parsed["date"] = msg.date
                new_messages.append(parsed)

        if new_messages:
            all_messages.extend(new_messages)
            all_messages.sort(key=lambda m: m["date"])

        now_utc = datetime.utcnow()
        now_tr = now_utc + timedelta(hours=3)
        dtstr_tr = now_tr.strftime("%d.%m.%Y %H:%M")
        dtstr_utc = now_utc.strftime("%Y-%m-%d %H:%M")

        per_coin, per_coin_xchain = analyze_all_periods(
            all_messages, now_utc.replace(tzinfo=timezone.utc))
        gunluk_hacim, hacim_error = safe_api_call(
            get_daily_volume_usd, max_retry=2, wait=5, coin="BTC")
        gunluk_fiyat, _ = safe_api_call(
            get_daily_price, max_retry=2, wait=5, coin="BTC")

        # Balina net akışı dinamik hesaplanıyor (son 1 saat)
        balina_1h = per_coin["BTC"][-2][1] if "BTC" in per_coin and len(
            per_coin["BTC"]) > 1 else {"in_amount": 0, "out_amount": 0}
        balina_net_1h = balina_1h["in_amount"] - \
            balina_1h["out_amount"] if balina_1h else 0

        if "BTC" in per_coin:
            whale_alert_raporu = format_btc_whale_report(
                per_coin["BTC"],
                per_coin_xchain["BTC"],
                gunluk_hacim,
                gunluk_fiyat,
                gunluk_hacim is not None,
                hacim_error,
                dtstr_tr
            )
        else:
            whale_alert_raporu = f"Tarih/Saat (TSI): {dtstr_tr}\n⚠️ Son transferlerde BTC hareketi yok."

        ohlcv_5m = get_spot_ohlcv("BTCUSDT", "5m", 50)
        ohlcv_15m = get_spot_ohlcv("BTCUSDT", "15m", 50)
        ohlcv_30m = get_spot_ohlcv("BTCUSDT", "30m", 50)
        ohlcv_1h = get_spot_ohlcv("BTCUSDT", "1h", 100)
        ohlcv_4h = get_spot_ohlcv("BTCUSDT", "4h", 60)
        ohlcv_1d = get_spot_ohlcv("BTCUSDT", "1d", 60)
        current_price = get_daily_price("BTC")[0] or 0

        kisavadeli_analiz_bolumu = btc_kisavadeli_analizler(
            {
                "5m": ohlcv_5m,
                "15m": ohlcv_15m,
                "30m": ohlcv_30m
            },
            current_price, dtstr_tr, dtstr_utc
        )

        ls_ratio_1h = get_long_short_ratio(period="1h") or 1.0

        rapor_1h, skor_1h, max_skor_1h, destek, direnç, ema7, ema21, macd_val, rsi_val, obv_val, trend, obv_1h_pct = btc_teknik_analiz_raporu(
            ohlcv_1h, current_price, dtstr_tr, dtstr_utc, balina_net_1h, ls_ratio_1h, vade="1 Saatlik Analiz"
        )
        rapor_4h, skor_4h, max_skor_4h, _, _, _, _, _, _, _, trend_4h, _ = btc_teknik_analiz_raporu(
            ohlcv_4h, current_price, dtstr_tr, dtstr_utc, balina_net_1h, ls_ratio_1h, vade="4 Saatlik Analiz"
        )
        rapor_1d, skor_1d, max_skor_1d, destek_d, direnç_d, ema7_d, ema21_d, macd_val_d, rsi_val_d, obv_val_d, trend_d, obv_1d_pct = btc_teknik_analiz_raporu(
            ohlcv_1d, current_price, dtstr_tr, dtstr_utc, balina_net_1h, ls_ratio_1h, vade="Günlük Analiz"
        )

        ks_5m = ks_15m = ks_30m = 0
nihai = nihai_oneri(
    skor_1h,
    skor_4h,
    skor_1d,
    balina_net_1h,
    ls_ratio_1h,
    obv_delta_1h,
    price_delta_1h
)

# 1) Grafik oluştur ve gönder
chart = plot_technical_indicators(ohlcv_1h)
chart_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
requests.post(
    chart_url, 
    data={"chat_id": TELEGRAM_CHAT_ID}, 
    files={"photo": chart}
)
# ... (önceki kodun burada)

        # 2) Backtest sonuçlarını ve dinamik yorumu ekle
win_rate, total_signals = backtest_strategy(ohlcv_1h)
dynamic_comment = generate_dynamic_comment(rsi_val, macd_val, obv_1h_pct)

msg = (
            kisavadeli_analiz_bolumu
            + rapor_1h
            + "\n─────\n" + rapor_4h
            + "\n─────\n" + rapor_1d
            + "\n─────\n"
            + nihai
            + f"\n💡 Uzman Yorumu: {dynamic_comment}"
            + f"\n🔍 Backtest Sonucu (Son {total_signals} Sinyal): %{win_rate:.1f} Başarı"
            + whale_alert_raporu
            + "\n\n" + btc_piyasa_analiz_turkce()
        )

async def main_loop():
    send_telegram_message_split(msg, max_len=4000)
    print(dtstr_tr, "--> Mesaj gönderildi.")

    gunluk_hacimler = {}
    gunluk_fiyatlar = {}
    for coin in COINGECKO_IDS:
        gunluk_hacim, _ = safe_api_call(
            get_daily_volume_usd, max_retry=2, wait=5, coin=coin)
        gunluk_fiyat, _ = safe_api_call(
            get_daily_price, max_retry=2, wait=5, coin=coin)
        gunluk_hacimler[coin] = gunluk_hacim
        gunluk_fiyatlar[coin] = gunluk_fiyat

    tum_balin_raporu = format_all_coins_whale_report(
        per_coin, per_coin_xchain, gunluk_hacimler, gunluk_fiyatlar, dtstr_tr
    )
    send_telegram_message_split(tum_balin_raporu, max_len=4000)

    await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main_loop())