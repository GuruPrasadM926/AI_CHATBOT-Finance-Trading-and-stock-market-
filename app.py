"""
FinBot — AI Trading & Stock Market Assistant
Free APIs: Google Gemini + Groq

Get free keys:
  Gemini → https://aistudio.google.com/app/apikey
  Groq   → https://console.groq.com

Run: streamlit run app.py
"""

import os, re, time, logging
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinBot — AI Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #070f1e; }
[data-testid="stSidebar"]          { background-color: #050c18; border-right:1px solid rgba(0,212,170,.15); }
[data-testid="stHeader"]           { background: transparent; }
[data-testid="stChatMessage"]      { border-radius:12px; margin-bottom:4px; }
[data-testid="stMetric"]           { background:rgba(0,212,170,.06); border:1px solid rgba(0,212,170,.18); border-radius:10px; padding:12px; }
[data-testid="stMetricValue"]      { font-size:22px !important; font-weight:700 !important; color:#e2e8f0 !important; }
[data-testid="stMetricLabel"]      { font-size:12px !important; color:#8896b3 !important; }
[data-testid="stChatInput"] textarea { background:#0d1e38; border:1px solid rgba(0,212,170,.3); border-radius:12px; color:#e2e8f0; }
.stTextInput input  { background:#0d1e38 !important; border:1px solid rgba(0,212,170,.25) !important; border-radius:8px !important; color:#e2e8f0 !important; }
.stButton > button  { background:linear-gradient(135deg,#00d4aa,#009982); border:none; border-radius:8px; color:#050c18; font-weight:700; }
hr { border-color:rgba(0,212,170,.12) !important; }
::-webkit-scrollbar       { width:4px; }
::-webkit-scrollbar-thumb { background:rgba(0,212,170,.3); border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are FinBot, a professional trading and stock market AI assistant.

VERY IMPORTANT RULE: If the user asks anything that is NOT related to trading, stocks, finance, investment, cryptocurrency, economics, or financial markets — you must respond with exactly:
"I'm FinBot, a trading and finance assistant. I can only help with questions related to stocks, trading, cryptocurrency, financial markets, and investing. Please ask me something related to finance!"

Do NOT answer questions about cooking, sports, movies, travel, technology unrelated to fintech, health, relationships, general knowledge, or any other non-finance topic.

Your expertise covers:
- Technical Analysis: RSI, MACD, Bollinger Bands, EMA/SMA, VWAP, ATR, Stochastic, Fibonacci, Ichimoku, chart patterns (head & shoulders, triangles, flags, wedges, cup & handle), candlestick patterns, support/resistance, divergence
- Fundamental Analysis: P/E, P/B, EV/EBITDA, DCF, earnings reports, EPS, revenue, margins, ROE, debt ratios, cash flow
- Trading Strategies: day trading, swing trading, scalping, momentum, mean reversion, breakout, trend following, pairs trading
- Options: calls, puts, spreads, iron condor, straddle/strangle, covered calls, LEAPS, Greeks (Delta, Gamma, Theta, Vega)
- Risk Management: position sizing, Kelly Criterion, stop losses, risk/reward, drawdown, portfolio diversification, hedging
- Market Structure: order types, market makers, Level 2, liquidity, dark pools, sector rotation, market cycles, VIX, sentiment
- Macroeconomics: Fed policy, interest rates, yield curve, inflation, GDP, NFP, dollar index impact
- Cryptocurrency: Bitcoin, Ethereum, DeFi, on-chain metrics, crypto cycles

When live market data is provided, use it directly in your analysis. Give specific, practical, educational answers using real ticker examples. Format responses with **bold**, ## headers, and bullet points. Educational content — not personalized financial advice."""

WELCOME = """**Welcome to FinBot** 📈 — AI Trading & Stock Market Assistant

Ask me anything about:
- **Technical Analysis** — RSI, MACD, chart patterns, indicators
- **Fundamental Analysis** — earnings, P/E, valuation, cash flow
- **Trading Strategies** — swing, day trading, options, momentum
- **Risk Management** — position sizing, stop losses, portfolio theory
- **Live Market Data** — real-time prices + technical indicators for any ticker
- **Crypto, Macro, Options Greeks** and much more

> *Educational only — not financial advice.*"""

SUGGESTIONS = [
    "📊 Explain RSI and how to use it",
    "📈 What is a bull flag pattern?",
    "🔍 How do I read an earnings report?",
    "⚙️ Explain options Greeks (Delta, Gamma, Theta)",
    "🔄 What is swing trading strategy?",
    "🏛️ How do Fed rates affect stocks?",
    "📉 Explain support and resistance",
    "🎯 How does MACD work?",
]

MODELS = {
    "gemini": {
        "gemini-2.0-flash":         "Gemini 2.0 Flash ⚡ (Recommended)",
        "gemini-1.5-flash":         "Gemini 1.5 Flash (Fast)",
        "gemini-1.5-pro":           "Gemini 1.5 Pro (Most Capable)",
    },
    "groq": {
        "llama-3.3-70b-versatile":  "Llama 3.3 70B (Best Quality)",
        "llama-3.1-8b-instant":     "Llama 3.1 8B (Ultra Fast)",
        "mixtral-8x7b-32768":       "Mixtral 8x7B (Great Balance)",
        "gemma2-9b-it":             "Gemma 2 9B (Google via Groq)",
    },
}

COMPANY_MAP = {
    "apple":"AAPL","microsoft":"MSFT","google":"GOOGL","alphabet":"GOOGL",
    "amazon":"AMZN","tesla":"TSLA","nvidia":"NVDA","meta":"META","facebook":"META",
    "netflix":"NFLX","amd":"AMD","intel":"INTC","jpmorgan":"JPM","goldman":"GS",
    "berkshire":"BRK-B","walmart":"WMT","disney":"DIS","uber":"UBER","airbnb":"ABNB",
    "palantir":"PLTR","coinbase":"COIN","shopify":"SHOP","visa":"V","mastercard":"MA",
    "paypal":"PYPL","salesforce":"CRM","adobe":"ADBE","oracle":"ORCL","ford":"F",
    "exxon":"XOM","chevron":"CVX","pfizer":"PFE","moderna":"MRNA","coca cola":"KO",
    "pepsi":"PEP","spotify":"SPOT","snap":"SNAP","roku":"ROKU","zoom":"ZM","rivian":"RIVN",
}

STOPWORDS = {
    "I","A","AN","AT","IN","ON","TO","BY","AS","OR","OF","BE","IS","IT","IF","DO","ME",
    "MY","HE","WE","US","UP","SO","NO","GO","AI","ML","AM","PM","UK","EU","VS","TV",
    "RSI","MACD","EPS","ROE","ROA","FCF","ATR","OBV","EMA","SMA","IPO","ETF","VIX",
    "GDP","CPI","FED","SEC","NOW","CEO","CFO","AND","THE","FOR","NOT","BUT","USD",
    "EUR","GBP","JPY","XRP","BTC","ETH","DID","HOW","WHY","WHO","CAN","MAY","WAY",
    "NEW","ALL","OUT","GET","PUT","YOU","ARE","WAS","HIS","HER","ITS",
}

# ── Market Data ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_svc():
    return MarketService()

class MarketService:
    def __init__(self):
        self._cache = {}
        self.news_key = os.getenv("NEWS_API_KEY", "")

    def _cached(self, key, fn, ttl=90):
        if key in self._cache:
            d, t = self._cache[key]
            if time.time() - t < ttl:
                return d
        r = fn()
        if r: self._cache[key] = (r, time.time())
        return r

    def stock_info(self, sym):
        def fetch():
            try:
                info = yf.Ticker(sym).info
                price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
                if not price:
                    h = yf.Ticker(sym).history(period="2d")
                    if h.empty: return {"error": f"No data for {sym}"}
                    price = float(h["Close"].iloc[-1])
                mc = info.get("marketCap") or 0
                mc_s = (f"${mc/1e12:.2f}T" if mc>=1e12 else f"${mc/1e9:.2f}B" if mc>=1e9
                        else f"${mc/1e6:.2f}M" if mc>=1e6 else "N/A")
                return {
                    "symbol": info.get("symbol", sym),
                    "name": info.get("longName") or info.get("shortName") or sym,
                    "price": float(price),
                    "change_pct": info.get("regularMarketChangePercent") or 0,
                    "day_high": info.get("dayHigh"), "day_low": info.get("dayLow"),
                    "volume": info.get("regularMarketVolume") or 0,
                    "avg_volume": info.get("averageVolume") or 0,
                    "market_cap": mc_s,
                    "pe_trailing": info.get("trailingPE"), "pe_forward": info.get("forwardPE"),
                    "eps": info.get("trailingEps"), "beta": info.get("beta"),
                    "week_52_high": info.get("fiftyTwoWeekHigh"),
                    "week_52_low": info.get("fiftyTwoWeekLow"),
                    "sector": info.get("sector"), "industry": info.get("industry"),
                    "profit_margin": info.get("profitMargins"),
                    "roe": info.get("returnOnEquity"),
                    "timestamp": datetime.now().strftime("%H:%M:%S"), "error": None,
                }
            except Exception as e:
                return {"error": str(e), "symbol": sym}
        return self._cached(f"i_{sym}", fetch)

    def technicals(self, sym):
        def fetch():
            try:
                h = yf.Ticker(sym).history(period="6mo", interval="1d")
                if h.empty or len(h) < 20: return {"error": "Not enough data"}
                c, hi, lo, v = h["Close"], h["High"], h["Low"], h["Volume"]
                n = len(c)
                ma20  = c.rolling(20).mean().iloc[-1]
                ma50  = c.rolling(min(50,n)).mean().iloc[-1]
                ma200 = c.rolling(min(200,n)).mean().iloc[-1]
                d = c.diff()
                rsi = float((100 - 100/(1 + d.clip(lower=0).rolling(14).mean() /
                             (-d.clip(upper=0)).rolling(14).mean().replace(0,np.nan))).iloc[-1])
                e12  = c.ewm(span=12,adjust=False).mean()
                e26  = c.ewm(span=26,adjust=False).mean()
                ml   = float((e12-e26).iloc[-1])
                sl   = float((e12-e26).ewm(span=9,adjust=False).mean().iloc[-1])
                bm   = float(c.rolling(20).mean().iloc[-1])
                bs   = float(c.rolling(20).std().iloc[-1])
                tr   = pd.concat([(hi-lo),(hi-c.shift()).abs(),(lo-c.shift()).abs()],axis=1).max(axis=1)
                atr  = float(tr.rolling(14).mean().iloc[-1])
                sk   = float((100*(c-lo.rolling(14).min())/(hi.rolling(14).max()-lo.rolling(14).min()).replace(0,np.nan)).rolling(3).mean().iloc[-1])
                vr   = float(v.iloc[-1]/(v.rolling(20).mean().iloc[-1] or 1))
                cur  = float(c.iloc[-1])
                return {
                    "symbol":sym,"current":cur,
                    "ma20":float(ma20),"ma50":float(ma50),"ma200":float(ma200),
                    "rsi":rsi,"rsi_sig":"Oversold" if rsi<30 else "Overbought" if rsi>70 else "Neutral",
                    "macd":ml,"macd_sig":sl,"macd_hist":ml-sl,
                    "macd_dir":"Bullish" if ml>sl else "Bearish",
                    "bb_upper":bm+2*bs,"bb_mid":bm,"bb_lower":bm-2*bs,
                    "bb_pos":("Near Upper" if cur>bm+2*bs else "Near Lower" if cur<bm-2*bs else "Mid-Band"),
                    "atr":atr,"stoch_k":sk,"vol_ratio":vr,
                    "above20":cur>ma20,"above50":cur>ma50,"above200":cur>ma200,"error":None,
                }
            except Exception as e:
                return {"error":str(e),"symbol":sym}
        return self._cached(f"t_{sym}", fetch, ttl=180)

    def news(self, sym=None, n=5):
        def fetch():
            if self.news_key:
                try:
                    q = sym or "stock market"
                    r = requests.get(f"https://newsapi.org/v2/everything?q={q}&language=en&sortBy=publishedAt&pageSize={n}&apiKey={self.news_key}",timeout=6).json()
                    if r.get("status")=="ok":
                        return [{"title":a["title"],"source":a["source"]["name"]} for a in r.get("articles",[])[:n] if a.get("title")]
                except: pass
            if sym:
                try:
                    raw = yf.Ticker(sym).news or []
                    return [{"title":x.get("title",""),"source":x.get("publisher","")} for x in raw[:n]]
                except: pass
            return []
        return self._cached(f"n_{sym}_{n}", fetch, ttl=300) or []


# ── Intent Classifier ─────────────────────────────────────────────────────────
class Intent:
    _KW = {
        "PRICE":    ["price","worth","trading at","how much","current price","quote"],
        "TECH":     ["rsi","macd","bollinger","moving average","technical","chart","indicator",
                     "support","resistance","trend","overbought","oversold","ema","sma","atr",
                     "stochastic","fibonacci","candlestick","pattern","volume"],
        "FUND":     ["p/e","pe ratio","earnings","revenue","profit","margin","valuation",
                     "fundamental","balance sheet","cash flow","eps","dividend","roe","debt","dcf"],
        "TRADE":    ["buy","sell","hold","short","long","invest","should i","trade","entry",
                     "exit","position","signal","worth buying"],
        "NEWS":     ["news","latest","update","happening","today","announcement","headline"],
        "OPTIONS":  ["option","call","put","strike","expiry","delta","gamma","theta","vega",
                     "iron condor","spread","covered call","straddle","strangle","implied volatility"],
        "CRYPTO":   ["bitcoin","ethereum","crypto","btc","eth","defi","blockchain","altcoin"],
        "MACRO":    ["fed","interest rate","inflation","gdp","cpi","yield curve","recession","nfp"],
        "RISK":     ["risk","stop loss","position size","portfolio","diversif","hedge","drawdown","kelly"],
    }
    _NEEDS_DATA = {"PRICE","TECH","FUND","TRADE","NEWS"}

    def classify(self, msg):
        m = msg.lower()
        sc = {i: sum(1 for k in kws if k in m) for i, kws in self._KW.items()}
        best = max(sc, key=sc.get)
        return best if sc[best] > 0 else "GENERAL"

    def needs_data(self, intent):
        return intent in self._NEEDS_DATA


# ── LLM Service (Gemini + Groq) ───────────────────────────────────────────────
class LLM:
    def __init__(self, provider, api_key, model):
        self.provider = provider
        self.api_key  = api_key.strip() if api_key else ""
        self.model    = model

    def chat(self, messages, market_ctx=""):
        if not self.api_key:
            return ("⚠️ **No API key set.**\n\n"
                    "Add your free key in the sidebar:\n"
                    "- **Gemini** → [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)\n"
                    "- **Groq** → [console.groq.com](https://console.groq.com)")

        # Inject market data into last user message
        msgs = []
        for i, m in enumerate(messages):
            if m["role"] not in ("user","assistant"): continue
            content = (f"[Live Market Data]\n{market_ctx}\n\n[Question]\n{m['content']}"
                       if i == len(messages)-1 and m["role"]=="user" and market_ctx
                       else m["content"])
            msgs.append({"role": m["role"], "content": content})

        try:
            return self._gemini(msgs) if self.provider == "gemini" else self._groq(msgs)
        except Exception as e:
            logger.error("LLM error: %s", e)
            return f"⚠️ **Error:** {str(e)[:300]}"
        
    def _groq(self, msgs):
        from groq import Groq
        r = Groq(api_key=self.api_key).chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":SYSTEM_PROMPT}] + msgs,
            max_tokens=1500, temperature=0.7,
        )
        return r.choices[0].message.content


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_symbols(text):
    found = []
    tl = text.lower()
    for co, tk in COMPANY_MAP.items():
        if co in tl and tk not in found: found.append(tk)
    for m in re.findall(r'\$([A-Za-z]{1,5})', text):
        t = m.upper()
        if t not in found: found.insert(0, t)
    for m in re.findall(r'\b[A-Z]{2,5}\b', text):
        if m not in STOPWORDS and m not in found: found.append(m)
    return found[:3]

def market_context(syms, intent, svc):
    if not syms: return ""
    lines = []
    for sym in syms[:2]:
        info = svc.stock_info(sym)
        if not info or info.get("error"):
            lines.append(f"\n[{sym}] Unavailable"); continue
        chg = info.get("change_pct") or 0
        lines += [f"\n{'─'*40}",
                  f"  {info.get('name',sym)} ({info['symbol']}) — {info.get('timestamp','')}",
                  f"{'─'*40}",
                  f"  Price      : ${info['price']:.2f}  ({chg:+.2f}% today)",
                  f"  Day Range  : ${info.get('day_low','N/A')} – ${info.get('day_high','N/A')}",
                  f"  52W Range  : ${info.get('week_52_low','N/A')} – ${info.get('week_52_high','N/A')}",
                  f"  Volume     : {info.get('volume',0):,}  (Avg {info.get('avg_volume',0):,})",
                  f"  Market Cap : {info.get('market_cap','N/A')}"]
        if info.get("pe_trailing"):    lines.append(f"  P/E TTM    : {info['pe_trailing']:.2f}")
        if info.get("pe_forward"):     lines.append(f"  P/E Fwd    : {info['pe_forward']:.2f}")
        if info.get("eps"):            lines.append(f"  EPS TTM    : ${info['eps']:.2f}")
        if info.get("beta"):           lines.append(f"  Beta       : {info['beta']:.2f}")
        if info.get("sector"):         lines.append(f"  Sector     : {info['sector']}")
        if info.get("profit_margin"):  lines.append(f"  Margin     : {info['profit_margin']*100:.1f}%")
        if info.get("roe"):            lines.append(f"  ROE        : {info['roe']*100:.1f}%")
        if intent in ("TECH","TRADE","FUND"):
            t = svc.technicals(sym)
            if t and not t.get("error"):
                lines += [f"\n  [Technicals — {sym}]",
                          f"  RSI(14)    : {t['rsi']:.1f}  [{t['rsi_sig']}]",
                          f"  MACD       : {t['macd']:.3f} / Signal {t['macd_sig']:.3f} / Hist {t['macd_hist']:.3f}  [{t['macd_dir']}]",
                          f"  MA 20/50/200: ${t['ma20']:.2f} / ${t['ma50']:.2f} / ${t['ma200']:.2f}",
                          f"  Bollinger  : ${t['bb_lower']:.2f} / {t['bb_mid']:.2f} / ${t['bb_upper']:.2f}  [{t['bb_pos']}]",
                          f"  ATR(14)    : ${t['atr']:.2f}  |  Stoch %K: {t['stoch_k']:.1f}  |  Vol Ratio: {t['vol_ratio']:.1f}x",
                          f"  vs MAs     : {'↑' if t['above20'] else '↓'}MA20  {'↑' if t['above50'] else '↓'}MA50  {'↑' if t['above200'] else '↓'}MA200"]
    return "\n".join(lines)

def metric_cards(syms, svc):
    valid = [svc.stock_info(s) for s in syms[:3]]
    valid = [i for i in valid if i and not i.get("error")]
    if not valid: return
    for col, info in zip(st.columns(len(valid)), valid):
        chg = info.get("change_pct") or 0
        col.metric(f"{info.get('name',info['symbol'])[:22]} ({info['symbol']})",
                   f"${info['price']:.2f}", f"{chg:+.2f}% today")


# ── Session init ──────────────────────────────────────────────────────────────
def init():
    prov = "groq"
    key  = os.getenv("GROQ_API_KEY") or ""
    for k,v in {
        "messages":        [{"role":"assistant","content":WELCOME}],
        "api_provider":    prov,
        "api_key":         key,
        "model":           list(MODELS[prov].keys())[0],
        "last_syms":       [],
        "show_suggestions":True,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar(svc):
    with st.sidebar:
        st.markdown("""<div style="text-align:center;padding:14px 0 8px">
        <div style="font-size:38px">📈</div>
        <div style="font-size:20px;font-weight:800;color:#e2e8f0">FinBot</div>
        <div style="font-size:10px;color:#00d4aa;font-family:monospace;letter-spacing:.1em">● LIVE · AI TRADING ASSISTANT</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        # ── Provider & model ──────────────────────────────────────────────
        st.markdown("### ⚙️ Settings")
        st.markdown("**Provider:** ⚡ Groq — Llama / Mixtral (Free)")
        st.session_state.api_provider = "groq"

        key = st.text_input("API Key", value=st.session_state.api_key,
                            type="password", placeholder="gsk_...")
        st.session_state.api_key = key

        mdl = st.selectbox("Model",
            options=list(MODELS["groq"].keys()),
            index=(list(MODELS["groq"]).index(st.session_state.model)
                   if st.session_state.model in MODELS["groq"] else 0),
            format_func=lambda m: MODELS["groq"].get(m, m))
        st.session_state.model = mdl

        if st.session_state.api_key:
            st.success("✅ API key set", icon="🔐")
        else:
            st.info("Get your free Groq key 👉 [console.groq.com](https://console.groq.com)")

        st.markdown("---")

        # ── Quick lookup ──────────────────────────────────────────────────
        st.markdown("### 🔍 Quick Lookup")
        sym = st.text_input("Ticker", placeholder="AAPL, TSLA, BTC-USD…").strip().upper()
        if sym:
            with st.spinner(f"Fetching {sym}…"):
                info = svc.stock_info(sym)
            if info and not info.get("error"):
                chg = info.get("change_pct") or 0
                st.metric(f"{info.get('name',sym)[:22]} ({sym})",
                          f"${info['price']:.2f}", f"{chg:+.2f}%")
                if info.get("sector"):
                    st.caption(f"📂 {info['sector']} › {info.get('industry','')}")
            else:
                st.error((info or {}).get("error","Not found"))

        st.markdown("---")

        # ── Market overview ───────────────────────────────────────────────
        st.markdown("### 📊 Market Overview")
        for s, lbl in {"SPY":"S&P 500","QQQ":"NASDAQ 100","DIA":"Dow Jones","^VIX":"VIX"}.items():
            info = svc.stock_info(s)
            if info and not info.get("error"):
                chg = info.get("change_pct") or 0
                st.markdown(f"{'🟢' if chg>=0 else '🔴'} **{lbl}** &nbsp;"
                            f"${info['price']:.2f} ({chg:+.2f}%)")

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages        = [{"role":"assistant","content":WELCOME}]
            st.session_state.last_syms       = []
            st.session_state.show_suggestions = True
            st.rerun()

        st.markdown("<div style='font-size:11px;color:#3a4a5a;text-align:center;padding-top:8px'>"
                    "Educational content only.<br>Not financial advice.</div>", unsafe_allow_html=True)


# ── Chat ──────────────────────────────────────────────────────────────────────
def chat(svc):
    clf = Intent()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="📈" if msg["role"]=="assistant" else "👤"):
            st.markdown(msg["content"])

    if (st.session_state.last_syms and st.session_state.messages
            and st.session_state.messages[-1]["role"]=="assistant"):
        metric_cards(st.session_state.last_syms, svc)

    if st.session_state.show_suggestions:
        st.markdown("**💡 Quick questions:**")
        cols = st.columns(2)
        for i, s in enumerate(SUGGESTIONS):
            if cols[i%2].button(s, key=f"s{i}", use_container_width=True):
                process(s.split(" ",1)[1], svc, clf); return

    if prompt := st.chat_input("Ask about stocks, strategies, technical analysis…"):
        process(prompt, svc, clf)

def process(prompt, svc, clf):
    st.session_state.show_suggestions = False
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user", avatar="👤"): st.markdown(prompt)

    intent = clf.classify(prompt)

    syms = []
    if clf.needs_data(intent):
        syms = extract_symbols(prompt)
        if not syms:
            for past in reversed([m["content"] for m in st.session_state.messages[-10:]
                                   if m["role"]=="user"][:-1]):
                syms = extract_symbols(past)
                if syms: break
    st.session_state.last_syms = syms

    ctx = ""
    if syms and clf.needs_data(intent):
        with st.spinner("📡 Fetching live market data…"):
            ctx = market_context(syms, intent, svc)

    llm = LLM(st.session_state.api_provider, st.session_state.api_key, st.session_state.model)
    with st.chat_message("assistant", avatar="📈"):
        with st.spinner("🤔 FinBot is analyzing…"):
            resp = llm.chat(st.session_state.messages, ctx)
        st.markdown(resp)
        if syms: metric_cards(syms, svc)

    st.session_state.messages.append({"role":"assistant","content":resp})


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init()
    svc = get_svc()
    sidebar(svc)
    chat(svc)

main()
