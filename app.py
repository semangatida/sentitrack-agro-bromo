import streamlit as st
import pandas as pd
import numpy as np
import re, time, warnings
warnings.filterwarnings('ignore')
from collections import Counter

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SentiTrack · Agro Bromo",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS: Modern Colorful Startup Dashboard ───────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #F0F4FF;
    --white: #FFFFFF;
    --card: #FFFFFF;
    --border: #E2E8F8;
    --primary: #4F46E5;
    --primary-light: #EEF2FF;
    --pink: #EC4899;
    --pink-light: #FDF2F8;
    --green: #10B981;
    --green-light: #ECFDF5;
    --amber: #F59E0B;
    --amber-light: #FFFBEB;
    --red: #EF4444;
    --red-light: #FEF2F2;
    --blue: #3B82F6;
    --blue-light: #EFF6FF;
    --text: #0F172A;
    --text2: #475569;
    --text3: #94A3B8;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(79,70,229,0.06);
    --shadow-hover: 0 4px 12px rgba(0,0,0,0.1), 0 8px 32px rgba(79,70,229,0.12);
}

*, body, html { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* Cards */
.card {
    background: var(--white);
    border-radius: 16px;
    padding: 24px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    transition: box-shadow 0.2s, transform 0.2s;
}
.card:hover { box-shadow: var(--shadow-hover); transform: translateY(-1px); }

/* Metric card */
.metric-card {
    background: var(--white);
    border-radius: 14px;
    padding: 20px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}
.metric-icon {
    width: 44px; height: 44px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; margin-bottom: 12px;
}
.metric-val {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 32px; font-weight: 700;
    color: var(--text); line-height: 1;
}
.metric-lbl { font-size: 13px; color: var(--text2); margin-top: 4px; }

/* Gradient header */
.page-header {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
    border-radius: 20px;
    padding: 32px 36px;
    color: white;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
}
.page-header::after {
    content: '';
    position: absolute; bottom: -60px; right: 80px;
    width: 150px; height: 150px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.header-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 28px; font-weight: 700;
    margin-bottom: 6px;
}
.header-sub { font-size: 14px; opacity: 0.85; }

/* Sentiment badges */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 99px;
    font-size: 12px; font-weight: 600;
}
.badge-pos { background: var(--green-light); color: var(--green); }
.badge-neu { background: var(--amber-light); color: var(--amber); }
.badge-neg { background: var(--red-light); color: var(--red); }

/* Nav item */
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 10px;
    font-size: 14px; font-weight: 500; color: var(--text2);
    cursor: pointer; transition: all 0.15s; margin-bottom: 2px;
}
.nav-item:hover { background: var(--primary-light); color: var(--primary); }
.nav-item.active { background: var(--primary-light); color: var(--primary); font-weight: 600; }

/* Score bar */
.score-wrap { margin-bottom: 12px; }
.score-row { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 5px; }
.score-track { background: #F1F5F9; border-radius: 99px; height: 8px; overflow: hidden; }
.score-fill { height: 100%; border-radius: 99px; transition: width 0.6s ease; }

/* Input */
.stTextArea textarea {
    border-radius: 12px !important;
    border: 2px solid var(--border) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 14px !important;
    transition: border-color 0.2s !important;
    background: var(--white) !important;
}
.stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
}
.stTextInput input {
    border-radius: 10px !important;
    border: 2px solid var(--border) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
}
.stTextInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important; font-size: 14px !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.4) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 2px solid var(--border) !important;
}

/* Tab */
button[data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 14px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary) !important;
}

/* Channel tags */
.ch-tvone   { background: #FEE2E2; color: #DC2626; padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; }
.ch-kompas  { background: #DBEAFE; color: #1D4ED8; padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; }
.ch-metro   { background: #D1FAE5; color: #059669; padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; }

/* Result highlight */
.result-big {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 42px; font-weight: 700; line-height: 1;
}

/* Step indicator */
.step-pill {
    background: var(--primary-light);
    color: var(--primary);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 12px; font-weight: 700;
    display: inline-block; margin-bottom: 8px;
}

/* Table override */
.stDataFrame { border-radius: 12px !important; overflow: hidden; }
[data-testid="stDataFrameResizable"] { border-radius: 12px !important; }

/* Progress */
.stProgress > div > div > div { background: var(--primary) !important; }

/* Info/warning */
.stAlert { border-radius: 12px !important; }

div[data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)

# ── NLP & Model Loaders ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_nltk():
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    return word_tokenize, stopwords

@st.cache_resource(show_spinner=False)
def load_roberta():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline("text-classification",
        model="w11wo/indonesian-roberta-base-sentiment-classifier",
        truncation=True, max_length=512, device=-1, top_k=None)

@st.cache_resource(show_spinner=False)
def load_indobert():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline("text-classification",
        model="mdhugol/indonesia-bert-sentiment-classification",
        truncation=True, max_length=512, device=-1, top_k=None)

# ── Slang Dict ───────────────────────────────────────────────
SLANG = {
    'gk':'tidak','ga':'tidak','gak':'tidak','nggak':'tidak','ngga':'tidak',
    'bgt':'sangat','banget':'sangat','bngt':'sangat',
    'gue':'saya','gw':'saya','aku':'saya','ak':'saya',
    'lo':'kamu','lu':'kamu','u':'kamu',
    'yg':'yang','dr':'dari','dgn':'dengan','aja':'saja',
    'klo':'kalau','kalo':'kalau','krn':'karena','karna':'karena',
    'jd':'jadi','udh':'sudah','udah':'sudah','blm':'belum',
    'lg':'lagi','jg':'juga','tp':'tapi','skrg':'sekarang',
    'emg':'memang','bener':'benar','bs':'bisa',
    'tau':'tahu','tw':'tahu','wkwk':'tertawa','haha':'tertawa',
    'nih':'ini','tuh':'itu','gini':'begini','gitu':'begitu',
}

LABEL_ROBERTA = {'positive':'positive','negative':'negative','neutral':'neutral',
                 'POSITIVE':'positive','NEGATIVE':'negative','NEUTRAL':'neutral'}
LABEL_INDOBERT = {'LABEL_0':'positive','LABEL_1':'neutral','LABEL_2':'negative',
                  'Positif':'positive','Netral':'neutral','Negatif':'negative'}

SENT_COLOR = {'positive':'#10B981','neutral':'#F59E0B','negative':'#EF4444'}
SENT_ID    = {'positive':'Positif','neutral':'Netral','negative':'Negatif'}
SENT_ICON  = {'positive':'😊','neutral':'😐','negative':'😔'}

CHANNEL_COLORS = {'TVONE':'#EF4444','KOMPAS':'#3B82F6','METROTV':'#10B981'}

# ── Text Processing ──────────────────────────────────────────
def clean_text(t):
    if pd.isna(t): return ''
    t = str(t).lower()
    t = re.sub(r'<[^>]+>',' ',t)
    t = re.sub(r'https?://\S+|www\.\S+',' ',t)
    t = re.sub(r'@\w+',' ',t)
    t = re.sub(r'[^\x00-\x7F]+',' ',t)
    t = re.sub(r'\d+',' ',t)
    t = re.sub(r'[^\w\s]',' ',t)
    t = re.sub(r'(.)\1{2,}',r'\1\1',t)
    return re.sub(r'\s+',' ',t).strip()

def normalize(t):
    return ' '.join(SLANG.get(w,w) for w in t.split())

def preprocess(t):
    return normalize(clean_text(t))

def analyze(text, model_name):
    processed = preprocess(text)
    if not processed.strip():
        return {'label':'neutral','score':1.0,'scores':{'positive':0.0,'neutral':1.0,'negative':0.0}}
    try:
        clf = load_roberta() if model_name == "RoBERTa" else load_indobert()
        lmap = LABEL_ROBERTA if model_name == "RoBERTa" else LABEL_INDOBERT
        results = clf(processed[:512])
        if isinstance(results[0], list): results = results[0]
        scores = {}
        for r in results:
            lbl = lmap.get(r['label'], r['label'].lower())
            scores[lbl] = r.get('score', 0.0)
        for k in ['positive','neutral','negative']:
            scores.setdefault(k, 0.0)
        total = sum(scores.values()) or 1
        scores = {k: v/total for k,v in scores.items()}
        label = max(scores, key=scores.get)
        return {'label': label, 'score': scores[label], 'scores': scores}
    except Exception as e:
        return {'label':'neutral','score':1.0,'scores':{'positive':0.0,'neutral':1.0,'negative':0.0},'error':str(e)}

# ── YouTube Crawler ──────────────────────────────────────────
def crawl_youtube(api_key, video_dict):
    try:
        from googleapiclient.discovery import build
    except ImportError:
        st.error("Install: `pip install google-api-python-client`")
        return pd.DataFrame()

    all_comments = []
    youtube = build('youtube', 'v3', developerKey=api_key)
    progress = st.progress(0)
    status   = st.empty()
    total_ch = len(video_dict)

    for idx, (source, vid_id) in enumerate(video_dict.items()):
        status.markdown(f'<div style="font-size:13px;color:#475569;">🔄 Crawling <b>{source}</b>...</div>', unsafe_allow_html=True)
        page_count = 0
        try:
            response = youtube.commentThreads().list(
                part='snippet,replies', videoId=vid_id,
                maxResults=100, order='time'
            ).execute()
        except Exception as e:
            st.warning(f"Error {source}: {e}")
            progress.progress((idx+1)/total_ch)
            continue

        while response:
            page_count += 1
            for item in response.get('items', []):
                s = item['snippet']['topLevelComment']['snippet']
                all_comments.append({
                    'Source': source, 'VideoID': vid_id,
                    'CommentID': item['snippet']['topLevelComment']['id'],
                    'Date': s['publishedAt'], 'UserName': s['authorDisplayName'],
                    'Comment': s['textDisplay'], 'Like': s['likeCount'],
                    'Type': 'MAIN', 'ReplyCount': item['snippet']['totalReplyCount']
                })
                if item['snippet']['totalReplyCount'] > 0 and 'replies' in item:
                    for r in item['replies']['comments']:
                        rs = r['snippet']
                        all_comments.append({
                            'Source': source, 'VideoID': vid_id,
                            'CommentID': r['id'], 'Date': rs['publishedAt'],
                            'UserName': rs['authorDisplayName'],
                            'Comment': rs['textDisplay'], 'Like': rs['likeCount'],
                            'Type': 'REPLY', 'ReplyCount': 0
                        })
            if 'nextPageToken' in response:
                time.sleep(0.3)
                try:
                    response = youtube.commentThreads().list(
                        part='snippet,replies', pageToken=response['nextPageToken'],
                        videoId=vid_id, maxResults=100, order='time'
                    ).execute()
                except: break
            else:
                break
        progress.progress((idx+1)/total_ch)

    status.empty()
    progress.empty()
    if not all_comments:
        return pd.DataFrame()

    df = pd.DataFrame(all_comments)
    df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce').dt.tz_localize(None)
    df['Like'] = pd.to_numeric(df['Like'], errors='coerce').fillna(0).astype(int)
    before = len(df)
    df.drop_duplicates(subset=['CommentID'], keep='first', inplace=True)
    df.drop_duplicates(subset=['Comment','UserName'], keep='first', inplace=True)
    df.dropna(subset=['Comment'], inplace=True)
    df = df[df['Comment'].str.strip().str.len() >= 5].reset_index(drop=True)
    return df

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 16px 12px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <div style="background:linear-gradient(135deg,#4F46E5,#EC4899);border-radius:10px;
                        width:36px;height:36px;display:flex;align-items:center;justify-content:center;
                        font-size:18px;">🚆</div>
            <div>
                <div style="font-weight:800;font-size:16px;color:#0F172A;">SentiTrack</div>
                <div style="font-size:11px;color:#94A3B8;">Agro Bromo Analysis</div>
            </div>
        </div>
    </div>
    <hr style="border:none;border-top:1px solid #E2E8F8;margin:0 16px 12px;">
    """, unsafe_allow_html=True)

    page = st.selectbox("", ["🏠  Beranda", "📡  Crawling YouTube", "🔬  Analisis Sentimen",
                              "📊  Dashboard Data", "🎯  Batch Prediksi"],
                        label_visibility="collapsed")

    st.markdown("<hr style='border:none;border-top:1px solid #E2E8F8;margin:12px 16px;'>", unsafe_allow_html=True)
    st.markdown('<div style="padding:0 16px;font-size:11px;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Pengaturan</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Model", ["RoBERTa", "IndoBERT"], label_visibility="collapsed")
    show_prep    = st.toggle("Tampilkan preprocessing", False)

    st.markdown("""
    <div style="padding:12px 16px;margin-top:8px;">
        <div style="background:#F0F4FF;border-radius:10px;padding:12px;">
            <div style="font-size:11px;font-weight:700;color:#4F46E5;margin-bottom:6px;">PIPELINE</div>
            <div style="font-size:12px;color:#475569;line-height:2;">
                ① Crawl YouTube API<br>② Clean + Normalize<br>③ Transformer Sentiment<br>④ Visualisasi & Insight
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# BERANDA
# ══════════════════════════════════════════════════════
if "Beranda" in page:
    st.markdown("""
    <div class="page-header">
        <div class="header-title">🚆 Web Mining & Analisis Sentimen</div>
        <div class="header-sub">Komentar YouTube · Kecelakaan Kereta Api Agro Bromo · TVONE · KOMPAS · METROTV</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    stats = [
        ("#4F46E5","#EEF2FF","🗂️","Data Source","3 Channel YouTube","TVONE, KOMPAS, METROTV"),
        ("#EC4899","#FDF2F8","🤖","Model AI","RoBERTa / IndoBERT","Indonesian fine-tuned"),
        ("#10B981","#ECFDF5","⚡","Fitur","Crawl Langsung","Input API Key → data real"),
        ("#F59E0B","#FFFBEB","🎯","Output","Sentimen + Insight","Positif · Netral · Negatif"),
    ]
    for col,(color,bg,icon,label,val,sub) in zip([c1,c2,c3,c4],stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon" style="background:{bg};color:{color};">{icon}</div>
                <div style="font-size:11px;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;">{label}</div>
                <div style="font-weight:700;font-size:16px;color:#0F172A;margin:4px 0;">{val}</div>
                <div style="font-size:12px;color:#94A3B8;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_left, c_right = st.columns([3,2])

    with c_left:
        st.markdown("""
        <div class="card">
            <div style="font-weight:700;font-size:16px;color:#0F172A;margin-bottom:16px;">📌 Rumusan Masalah</div>
        """, unsafe_allow_html=True)
        rms = [
            ("#4F46E5","RM 1","Distribusi Sentimen","Bagaimana distribusi sentimen netizen terhadap pemberitaan kecelakaan KA Agro Bromo?"),
            ("#EC4899","RM 2","Pola per Channel","Adakah perbedaan pola sentimen antara TVONE, KOMPAS, dan METROTV?"),
            ("#10B981","RM 3","Performa ML","Bagaimana performa Random Forest & XGBoost dalam klasifikasi sentimen?"),
        ]
        for color,rm,title,desc in rms:
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:14px;">
                <div style="background:{color}15;color:{color};border-radius:8px;padding:6px 10px;
                            font-weight:700;font-size:12px;white-space:nowrap;min-width:44px;text-align:center;">{rm}</div>
                <div>
                    <div style="font-weight:600;font-size:14px;color:#0F172A;">{title}</div>
                    <div style="font-size:13px;color:#64748B;margin-top:2px;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown("""
        <div class="card">
            <div style="font-weight:700;font-size:16px;color:#0F172A;margin-bottom:16px;">🎬 Channel YouTube</div>
        """, unsafe_allow_html=True)
        channels = [
            ("#EF4444","#FEE2E2","TVONE","qHr-ky9Iwik"),
            ("#1D4ED8","#DBEAFE","KOMPAS","-lvgdiR6z1g"),
            ("#059669","#D1FAE5","METROTV","5EHTgRAyyMw"),
        ]
        for color,bg,name,vid in channels:
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        background:{bg};border-radius:10px;padding:12px 14px;margin-bottom:8px;">
                <div>
                    <div style="font-weight:700;font-size:14px;color:{color};">{name}</div>
                    <div style="font-size:11px;color:#64748B;font-family:monospace;">{vid}</div>
                </div>
                <div style="font-size:20px;">📺</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# CRAWLING
# ══════════════════════════════════════════════════════
elif "Crawling" in page:
    st.markdown("""
    <div class="page-header" style="background:linear-gradient(135deg,#0EA5E9,#4F46E5,#7C3AED);">
        <div class="header-title">📡 Crawling YouTube</div>
        <div class="header-sub">Input API Key & Video ID → ambil komentar langsung dari YouTube</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ Cara mendapatkan YouTube API Key", expanded=False):
        st.markdown("""
        1. Buka [console.cloud.google.com](https://console.cloud.google.com)
        2. Buat project baru → Enable **YouTube Data API v3**
        3. Credentials → Create API Key → copy
        """)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700;font-size:15px;color:#0F172A;margin-bottom:16px;">🔑 Konfigurasi API</div>', unsafe_allow_html=True)

    api_key = st.text_input("YouTube Data API Key", type="password",
                             placeholder="AIzaSy...", help="API key dari Google Cloud Console")

    st.markdown('<div style="font-weight:600;font-size:14px;color:#0F172A;margin:16px 0 10px;">🎬 Video ID per Channel</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        vid_tvone   = st.text_input("TVONE Video ID",   value="qHr-ky9Iwik")
    with c2:
        vid_kompas  = st.text_input("KOMPAS Video ID",  value="-lvgdiR6z1g")
    with c3:
        vid_metro   = st.text_input("METROTV Video ID", value="5EHTgRAyyMw")

    st.markdown('<br>', unsafe_allow_html=True)
    col_btn, col_info = st.columns([1,3])
    with col_btn:
        crawl_btn = st.button("🚀 Mulai Crawling", use_container_width=True)
    with col_info:
        st.markdown('<div style="font-size:12px;color:#94A3B8;padding-top:12px;">Proses crawling mungkin memakan waktu 2–5 menit tergantung jumlah komentar.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if crawl_btn:
        if not api_key.strip():
            st.error("Masukkan API Key terlebih dahulu!")
        else:
            videos = {}
            if vid_tvone.strip():   videos['TVONE']   = vid_tvone.strip()
            if vid_kompas.strip():  videos['KOMPAS']  = vid_kompas.strip()
            if vid_metro.strip():   videos['METROTV'] = vid_metro.strip()

            if not videos:
                st.warning("Masukkan minimal 1 Video ID.")
            else:
                with st.spinner("Sedang crawling..."):
                    df = crawl_youtube(api_key, videos)

                if df.empty:
                    st.error("Tidak ada data terkumpul. Cek API Key dan Video ID.")
                else:
                    st.session_state['df_raw'] = df
                    st.success(f"✅ Berhasil! **{len(df):,} komentar** dari {len(videos)} channel.")

                    c1,c2,c3 = st.columns(3)
                    for col, (src, cnt) in zip([c1,c2,c3], df['Source'].value_counts().items()):
                        with col:
                            color = CHANNEL_COLORS.get(src,'#4F46E5')
                            st.markdown(f"""
                            <div class="metric-card" style="border-top:3px solid {color};">
                                <div style="font-weight:700;color:{color};font-size:13px;">{src}</div>
                                <div class="metric-val">{cnt:,}</div>
                                <div class="metric-lbl">komentar</div>
                            </div>""", unsafe_allow_html=True)

                    st.markdown('<br>', unsafe_allow_html=True)
                    st.dataframe(df.head(20), use_container_width=True, height=250)

                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download CSV", csv,
                                       file_name="komentar_agro_bromo.csv", mime="text/csv")

    elif 'df_raw' in st.session_state:
        df = st.session_state['df_raw']
        st.info(f"✅ Data tersimpan: **{len(df):,} komentar**. Lanjut ke halaman lain atau crawl ulang.")

# ══════════════════════════════════════════════════════
# ANALISIS SENTIMEN
# ══════════════════════════════════════════════════════
elif "Analisis" in page:
    st.markdown("""
    <div class="page-header" style="background:linear-gradient(135deg,#7C3AED,#EC4899,#F59E0B);">
        <div class="header-title">🔬 Analisis Sentimen</div>
        <div class="header-sub">Masukkan komentar YouTube untuk dianalisis secara real-time</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick examples
    st.markdown('<div style="font-size:12px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Coba Contoh Komentar</div>', unsafe_allow_html=True)
    e1,e2,e3 = st.columns(3)
    examples = [
        ("😔 Negatif","Ini sungguh tragedi yang menyedihkan! Pemerintah harus segera memperbaiki sistem keselamatan kereta api kita."),
        ("😐 Netral","Saya lihat dari laporan Metro TV, sepertinya ada masalah teknis pada sistem pengereman. Perlu investigasi lebih lanjut."),
        ("🙏 Positif","Terima kasih tim penyelamat yang sudah bekerja keras. Semoga para korban cepat pulih dan bisa berkumpul dengan keluarga."),
    ]
    for col,(lbl,txt) in zip([e1,e2,e3],examples):
        with col:
            if st.button(lbl, use_container_width=True):
                st.session_state['input_text'] = txt

    st.markdown('<br>', unsafe_allow_html=True)

    col_in, col_out = st.columns([1,1])
    with col_in:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        user_input = st.text_area("",
            value=st.session_state.get('input_text',''),
            height=160,
            placeholder="Ketik atau tempel komentar YouTube di sini...",
            label_visibility="collapsed", key="main_input")

        if user_input:
            wc = len(user_input.split())
            st.markdown(f'<div style="font-size:12px;color:#94A3B8;margin-bottom:8px;">{wc} kata · {len(user_input)} karakter</div>', unsafe_allow_html=True)

        analyze_btn = st.button("✨ Analisis Sekarang", use_container_width=True)

        if show_prep and user_input:
            st.markdown('<div style="font-size:12px;font-weight:600;color:#4F46E5;margin-top:12px;">Preprocessing</div>', unsafe_allow_html=True)
            st.code(f"Clean : {clean_text(user_input)[:100]}\nNorm  : {normalize(clean_text(user_input))[:100]}", language=None)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_out:
        st.markdown('<div class="card" style="min-height:280px;">', unsafe_allow_html=True)

        if analyze_btn and user_input.strip():
            with st.spinner("Menganalisis..."):
                result = analyze(user_input, model_choice)

            label  = result['label']
            score  = result['score']
            scores = result['scores']
            color  = SENT_COLOR.get(label,'#888')

            st.markdown(f"""
            <div style="text-align:center;padding:8px 0 16px;">
                <div style="font-size:52px;">{SENT_ICON.get(label,'🤔')}</div>
                <div class="result-big" style="color:{color};">{SENT_ID.get(label,label)}</div>
                <div style="font-size:13px;color:#64748B;margin-top:4px;">Konfiden: <b>{score*100:.1f}%</b> · {model_choice}</div>
            </div>
            <hr style="border:none;border-top:1px solid #E2E8F8;margin:8px 0 16px;">
            """, unsafe_allow_html=True)

            for sk,sl in [('positive','Positif'),('neutral','Netral'),('negative','Negatif')]:
                s = scores.get(sk,0)*100
                c = SENT_COLOR[sk]
                st.markdown(f"""
                <div class="score-wrap">
                    <div class="score-row">
                        <span style="color:#475569;font-weight:500;">{sl}</span>
                        <span style="color:{c};font-weight:700;">{s:.1f}%</span>
                    </div>
                    <div class="score-track">
                        <div class="score-fill" style="width:{s}%;background:{c};"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        elif analyze_btn:
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            st.markdown("""
            <div style="text-align:center;padding:40px 20px;color:#CBD5E1;">
                <div style="font-size:48px;margin-bottom:12px;">🎯</div>
                <div style="font-size:14px;">Hasil analisis akan muncul di sini</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# DASHBOARD DATA
# ══════════════════════════════════════════════════════
elif "Dashboard" in page:
    st.markdown("""
    <div class="page-header" style="background:linear-gradient(135deg,#10B981,#0EA5E9,#4F46E5);">
        <div class="header-title">📊 Dashboard Data</div>
        <div class="header-sub">Visualisasi distribusi sentimen dan pola komentar per channel</div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = st.session_state.get('df_raw', None)
    uploaded = st.file_uploader("Upload CSV (atau gunakan data hasil crawling)", type=["csv"])
    if uploaded:
        import io
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        st.session_state['df_raw'] = df

    if df is not None and 'sentiment' not in df.columns:
        st.info("Data belum punya kolom sentimen. Jalankan batch prediksi dulu, atau upload CSV yang sudah ada kolom 'sentiment'.")

    if df is not None and 'sentiment' in df.columns:
        # Summary metrics
        total = len(df)
        sent_counts = df['sentiment'].value_counts()
        c1,c2,c3,c4 = st.columns(4)
        metric_data = [
            ("#4F46E5","#EEF2FF","📝","Total","komentar",f"{total:,}"),
            ("#10B981","#ECFDF5","😊","Positif","komentar",f"{sent_counts.get('positive',0):,}"),
            ("#F59E0B","#FFFBEB","😐","Netral","komentar",f"{sent_counts.get('neutral',0):,}"),
            ("#EF4444","#FEF2F2","😔","Negatif","komentar",f"{sent_counts.get('negative',0):,}"),
        ]
        for col,(color,bg,icon,lbl,sub,val) in zip([c1,c2,c3,c4],metric_data):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon" style="background:{bg};color:{color};">{icon}</div>
                    <div class="metric-val">{val}</div>
                    <div class="metric-lbl">{lbl} {sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams.update({
                'figure.facecolor':'#FFFFFF','axes.facecolor':'#FFFFFF',
                'text.color':'#0F172A','axes.labelcolor':'#475569',
                'xtick.color':'#64748B','ytick.color':'#64748B',
                'axes.spines.top':False,'axes.spines.right':False,
                'axes.edgecolor':'#E2E8F8','grid.color':'#F1F5F9',
                'axes.grid':True,'grid.alpha':1,'font.family':'sans-serif',
            })

            COLORS = {'positive':'#10B981','neutral':'#F59E0B','negative':'#EF4444'}
            sent_order = ['positive','neutral','negative']

            fig, axes = plt.subplots(1,3,figsize=(15,4.5))
            fig.patch.set_facecolor('#FFFFFF')

            # 1. Bar - distribusi
            vals = [sent_counts.get(s,0) for s in sent_order]
            colors = [COLORS[s] for s in sent_order]
            bars = axes[0].bar(['Positif','Netral','Negatif'], vals,
                               color=colors, edgecolor='white', linewidth=2, width=0.5)
            for bar,v in zip(bars,vals):
                pct = v/total*100
                axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+total*0.01,
                             f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold', color='#0F172A')
            axes[0].set_title('Distribusi Sentimen', fontweight='bold', fontsize=13, pad=12)
            axes[0].set_ylabel('Jumlah Komentar', color='#64748B', fontsize=11)

            # 2. Stacked - per channel
            if 'Source' in df.columns:
                ct = df.groupby(['Source','sentiment']).size().unstack(fill_value=0)
                ct_pct = ct.div(ct.sum(axis=1),axis=0)*100
                bottom = np.zeros(len(ct_pct))
                for s in sent_order:
                    if s in ct_pct.columns:
                        axes[1].bar(ct_pct.index, ct_pct[s].values, bottom=bottom,
                                    label=s.capitalize(), color=COLORS[s], alpha=0.88, edgecolor='white', linewidth=1)
                        for i,(v,b) in enumerate(zip(ct_pct[s].values,bottom)):
                            if v > 5:
                                axes[1].text(i, b+v/2, f'{v:.0f}%', ha='center', va='center',
                                             fontsize=10, fontweight='bold', color='white')
                        bottom += ct_pct[s].values
                axes[1].set_title('Sentimen per Channel', fontweight='bold', fontsize=13, pad=12)
                axes[1].legend(fontsize=9, framealpha=0)
                axes[1].set_ylim(0,115)

            # 3. Pie
            wedge_colors = [COLORS[s] for s in sent_order if sent_counts.get(s,0) > 0]
            wedge_vals   = [sent_counts.get(s,0) for s in sent_order if sent_counts.get(s,0) > 0]
            wedge_labels = [s.capitalize() for s in sent_order if sent_counts.get(s,0) > 0]
            axes[2].pie(wedge_vals, labels=wedge_labels, colors=wedge_colors,
                        autopct='%1.1f%%', startangle=90,
                        wedgeprops={'edgecolor':'white','linewidth':3},
                        textprops={'fontsize':11})
            axes[2].set_title('Proporsi Sentimen', fontweight='bold', fontsize=13, pad=12)

            plt.tight_layout(pad=2)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error visualisasi: {e}")

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div style="font-weight:700;font-size:15px;color:#0F172A;margin-bottom:12px;">📋 Preview Data</div>', unsafe_allow_html=True)
        st.dataframe(df.head(30), use_container_width=True, height=280)

    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:60px;">
            <div style="font-size:52px;margin-bottom:16px;">📊</div>
            <div style="font-weight:700;font-size:18px;color:#0F172A;margin-bottom:8px;">Belum ada data</div>
            <div style="font-size:14px;color:#64748B;">Crawl dari YouTube atau upload CSV hasil analisis</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# BATCH PREDIKSI
# ══════════════════════════════════════════════════════
elif "Batch" in page:
    st.markdown("""
    <div class="page-header" style="background:linear-gradient(135deg,#F59E0B,#EF4444,#EC4899);">
        <div class="header-title">🎯 Batch Prediksi</div>
        <div class="header-sub">Analisis sentimen massal — dari data crawling atau input manual</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Input Manual", "📁 Upload / Data Crawling"])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        bulk = st.text_area("Komentar (satu baris = satu komentar):", height=180,
                            placeholder="Komentar pertama...\nKomentar kedua...\nDst...",
                            label_visibility="collapsed")
        c1,c2 = st.columns([2,1])
        with c1:
            max_n = st.slider("Maks komentar", 5, 50, 15)
        with c2:
            st.markdown('<br>', unsafe_allow_html=True)
            btn = st.button("🚀 Analisis Batch", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if btn and bulk.strip():
            lines = [l.strip() for l in bulk.strip().split('\n') if l.strip()][:max_n]
            prog = st.progress(0)
            results = []
            for i,line in enumerate(lines):
                r = analyze(line, model_choice)
                results.append({'Komentar': line[:70]+('…' if len(line)>70 else ''),
                                 'Sentimen': SENT_ID.get(r['label'],r['label']),
                                 'Konfiden': f"{r['score']*100:.1f}%",
                                 'Positif':  f"{r['scores'].get('positive',0)*100:.1f}%",
                                 'Netral':   f"{r['scores'].get('neutral',0)*100:.1f}%",
                                 'Negatif':  f"{r['scores'].get('negative',0)*100:.1f}%"})
                prog.progress((i+1)/len(lines))
            prog.empty()

            df_r = pd.DataFrame(results)
            st.dataframe(df_r, use_container_width=True)

            # Mini summary
            vc = df_r['Sentimen'].value_counts()
            c1,c2,c3 = st.columns(3)
            for col,(sk,sl,ico,color,bg) in zip([c1,c2,c3],[
                ('Positif','Positif','😊','#10B981','#ECFDF5'),
                ('Netral','Netral','😐','#F59E0B','#FFFBEB'),
                ('Negatif','Negatif','😔','#EF4444','#FEF2F2')]):
                with col:
                    cnt = vc.get(sk,0)
                    pct = cnt/len(df_r)*100 if df_r is not None and len(df_r) > 0 else 0
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center;border-top:3px solid {color};">
                        <div style="font-size:28px;">{ico}</div>
                        <div style="font-weight:700;color:{color};font-size:14px;">{sl}</div>
                        <div class="metric-val">{cnt}</div>
                        <div class="metric-lbl">{pct:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

            st.download_button("⬇️ Download Hasil", df_r.to_csv(index=False),
                               file_name="batch_result.csv", mime="text/csv")

    with tab2:
        df_crawl = st.session_state.get('df_raw', None)
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")

        if uploaded:
            import io
            df_crawl = pd.read_csv(io.BytesIO(uploaded.getvalue()))

        if df_crawl is not None:
            comment_col = next((c for c in df_crawl.columns if c.lower() in ['comment','komentar','text','teks']), None)
            if comment_col:
                st.success(f"✅ {len(df_crawl):,} baris · kolom: `{comment_col}`")
                n = st.slider("Jumlah baris diproses", 10, min(500, len(df_crawl)), min(100, len(df_crawl)))

                if st.button("🚀 Jalankan Prediksi", use_container_width=True):
                    sample = df_crawl[comment_col].dropna().head(n).tolist()
                    prog = st.progress(0)
                    labels, confs = [], []
                    for idx,text in enumerate(sample):
                        r = analyze(str(text), model_choice)
                        labels.append(r['label'])
                        confs.append(f"{r['score']*100:.1f}%")
                        prog.progress((idx+1)/len(sample))
                    prog.empty()

                    df_out = df_crawl.head(n).copy()
                    df_out['sentiment']  = labels
                    df_out['confidence'] = confs
                    st.session_state['df_raw'] = df_out

                    st.success(f"✅ Selesai! {len(df_out):,} komentar sudah diberi label.")
                    st.dataframe(df_out.head(30), use_container_width=True, height=280)
                    st.download_button("⬇️ Download CSV", df_out.to_csv(index=False),
                                       file_name="sentiment_labeled.csv", mime="text/csv")
            else:
                st.warning("Kolom 'comment', 'komentar', atau 'text' tidak ditemukan di CSV.")
        else:
            st.info("Crawl data dari tab 📡 Crawling YouTube terlebih dahulu, atau upload CSV di sini.")

