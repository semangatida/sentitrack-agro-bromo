import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import warnings
warnings.filterwarnings('ignore')

from collections import Counter

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="SentiTrack · Agro Bromo",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');

:root {
    --bg:       #F7F8FC;
    --surface:  #FFFFFF;
    --border:   #E4E6EF;
    --border2:  #C9CCE0;

    --indigo:   #4F46E5;
    --indigo2:  #6366F1;
    --teal:     #0EA5A0;
    --coral:    #F25C54;
    --amber:    #F59E0B;

    --positive: #059669;
    --neutral:  #D97706;
    --negative: #DC2626;

    --text:     #111827;
    --text2:    #374151;
    --muted:    #6B7280;
    --subtle:   #9CA3AF;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
}

.stApp { background: var(--bg); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 12px rgba(79,70,229,0.04);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* ── Metric cards ── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    transition: box-shadow 0.2s, border-color 0.2s;
}
.metric-card:hover {
    box-shadow: 0 4px 20px rgba(79,70,229,0.08);
    border-color: var(--border2);
}
.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--subtle);
    margin-bottom: 6px;
    font-weight: 600;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 34px;
    font-weight: 700;
    line-height: 1;
    color: var(--text);
}
.metric-sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 5px;
}

/* ── Sentiment badges ── */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-positive { background: #D1FAE5; color: #065F46; }
.badge-neutral  { background: #FEF3C7; color: #92400E; }
.badge-negative { background: #FEE2E2; color: #991B1B; }

/* ── Section title ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--text);
    margin: 28px 0 12px;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 20px 0;
}

/* ── Input ── */
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}
.stTextArea textarea:focus {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
}

/* ── Select ── */
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}

/* ── Button ── */
.stButton > button {
    background: var(--indigo) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: all 0.15s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: var(--indigo2) !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Progress ── */
.stProgress > div > div > div { background: var(--indigo) !important; }

/* ── Tables ── */
.stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }

/* ── Hero ── */
.hero-wrap { padding: 36px 0 16px; }
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 48px;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: var(--text);
}
.hero-accent { color: var(--indigo); }
.hero-sub {
    font-size: 15px;
    color: var(--muted);
    max-width: 520px;
    line-height: 1.7;
    margin-top: 10px;
}

/* ── Result card ── */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    margin-top: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}

/* ── Score bars ── */
.score-bar-container {
    background: #F3F4F6;
    border-radius: 99px;
    height: 6px;
    margin: 6px 0 14px;
    overflow: hidden;
}
.score-bar { height: 100%; border-radius: 99px; transition: width 0.8s ease; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--indigo) !important; }

/* ── Tabs ── */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--indigo) !important;
    border-bottom-color: var(--indigo) !important;
}

/* ── Slider ── */
.stSlider > div { color: var(--text) !important; }

/* ── Alert ── */
.stAlert { border-radius: 10px !important; }

/* ── Channel tags ── */
.channel-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.tag-tvone   { background: #FEE2E2; color: #B91C1C; }
.tag-kompas  { background: #DBEAFE; color: #1D4ED8; }
.tag-metrotv { background: #D1FAE5; color: #065F46; }

/* ── Sidebar nav pill ── */
.nav-section {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--subtle) !important;
    font-weight: 700;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── NLP Setup ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_nlp():
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    return word_tokenize, stopwords

@st.cache_resource(show_spinner=False)
def load_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    return factory.create_stemmer()

@st.cache_resource(show_spinner=False)
def load_roberta():
    from transformers import pipeline as hf_pipeline
    model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
    clf = hf_pipeline(
        "text-classification", model=model_name, tokenizer=model_name,
        truncation=True, max_length=512, device=-1,
    )
    return clf

@st.cache_resource(show_spinner=False)
def load_indobert():
    from transformers import pipeline as hf_pipeline
    model_name = "mdhugol/indonesia-bert-sentiment-classification"
    clf = hf_pipeline(
        "text-classification", model=model_name, tokenizer=model_name,
        truncation=True, max_length=512, device=-1,
    )
    return clf


# ── Constants ────────────────────────────────────────────────
SLANG_DICT = {
    'gk':'tidak','ga':'tidak','gak':'tidak','nggak':'tidak','ngga':'tidak',
    'gakk':'tidak','gaada':'tidak ada','tak':'tidak','tdk':'tidak',
    'bgt':'sangat','banget':'sangat','bngt':'sangat','parah':'sangat',
    'gue':'saya','gw':'saya','aku':'saya','ak':'saya','sy':'saya',
    'lo':'kamu','lu':'kamu','u':'kamu',
    'yg':'yang','dr':'dari','dgn':'dengan','aja':'saja',
    'klo':'kalau','kl':'kalau','kalo':'kalau',
    'krn':'karena','karna':'karena','jd':'jadi',
    'udh':'sudah','udah':'sudah','dah':'sudah',
    'blm':'belum','lg':'lagi','jg':'juga','tp':'tapi',
    'skrg':'sekarang','ntar':'nanti',
    'emg':'memang','bener':'benar','org':'orang',
    'bs':'bisa','hrs':'harus',
    'knp':'kenapa','gmn':'gimana',
    'wkwk':'tertawa','haha':'tertawa','hehe':'tertawa',
    'tau':'tahu','tw':'tahu',
    'nih':'ini','tuh':'itu','gini':'begini','gitu':'begitu',
}

SENTIMENT_COLORS = {
    'positive': '#059669',
    'neutral':  '#D97706',
    'negative': '#DC2626',
}

LABEL_MAP_ROBERTA = {
    'positive': 'positive', 'POSITIVE': 'positive', 'Positive': 'positive',
    'negative': 'negative', 'NEGATIVE': 'negative', 'Negative': 'negative',
    'neutral':  'neutral',  'NEUTRAL':  'neutral',  'Neutral':  'neutral',
}
LABEL_MAP_INDOBERT = {
    'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative',
    'Positif': 'positive',  'Netral': 'neutral',  'Negatif': 'negative',
    'positive': 'positive', 'neutral': 'neutral', 'negative': 'negative',
}


# ── Text Processing ──────────────────────────────────────────
def clean_text(text):
    if pd.isna(text): return ''
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_slang(text):
    return ' '.join(SLANG_DICT.get(w, w) for w in text.split())

def preprocess(text):
    cleaned = clean_text(text)
    normalized = normalize_slang(cleaned)
    return normalized

def analyze_sentiment(text, model_name, clf_roberta, clf_indobert):
    processed = preprocess(text)
    if not processed.strip():
        return {'label': 'neutral', 'score': 1.0, 'scores': {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}}

    try:
        if model_name == "RoBERTa":
            result = clf_roberta(processed[:512])[0]
            label = LABEL_MAP_ROBERTA.get(result['label'], result['label'])
            score = result['score']
            all_results = clf_roberta(processed[:512], top_k=None) if hasattr(clf_roberta, '__call__') else [result]
            scores = {}
            if isinstance(all_results, list) and len(all_results) > 1:
                for r in all_results:
                    mapped = LABEL_MAP_ROBERTA.get(r['label'], r['label'])
                    scores[mapped] = r.get('score', 0.0)
            else:
                scores = {label: score, **{k: (1-score)/2 for k in ['positive','neutral','negative'] if k != label}}
        else:
            result = clf_indobert(processed[:512])[0]
            label = LABEL_MAP_INDOBERT.get(result['label'], result['label'])
            score = result['score']
            scores = {label: score, **{k: (1-score)/2 for k in ['positive','neutral','negative'] if k != label}}

        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k,v in scores.items()}
        for key in ['positive','neutral','negative']:
            scores.setdefault(key, 0.0)

        return {'label': label, 'score': score, 'scores': scores}

    except Exception as e:
        return {'label': 'neutral', 'score': 1.0,
                'scores': {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}, 'error': str(e)}


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 16px 0 24px;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
            <div style="width:32px; height:32px; background: linear-gradient(135deg, #4F46E5, #0EA5A0);
                        border-radius:8px; display:flex; align-items:center; justify-content:center;
                        font-family:'Syne',sans-serif; font-weight:800; color:#fff; font-size:14px;">ST</div>
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:16px; color:#111827;">SentiTrack</div>
        </div>
        <div style="font-size:12px; color:#9CA3AF; padding-left:42px;">KA Agro Bromo</div>
    </div>
    <hr style="border:none; border-top:1px solid #E4E6EF; margin: 0 0 20px;">
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Navigasi</div>', unsafe_allow_html=True)
    page = st.selectbox(
        "Navigasi",
        ["Dashboard", "Analisis Sentimen", "Statistik & EDA", "Batch Prediksi", "Tentang"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border:none; border-top:1px solid #E4E6EF; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="nav-section">Konfigurasi Model</div>', unsafe_allow_html=True)

    model_choice = st.selectbox("Model Sentimen", ["RoBERTa", "IndoBERT"], key="model_sel")
    show_preprocessing = st.toggle("Tampilkan Preprocessing", value=False)

    st.markdown("<hr style='border:none; border-top:1px solid #E4E6EF; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px; color:#6B7280; line-height:2.2;">
        <div class="nav-section" style="margin-bottom:8px;">Pipeline</div>
        <div style="display:flex; flex-direction:column; gap:4px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:20px; height:20px; background:#EEF2FF; border-radius:5px;
                            font-size:10px; font-weight:700; color:#4F46E5; display:flex;
                            align-items:center; justify-content:center;">1</div>
                <span>Web Crawling</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:20px; height:20px; background:#EEF2FF; border-radius:5px;
                            font-size:10px; font-weight:700; color:#4F46E5; display:flex;
                            align-items:center; justify-content:center;">2</div>
                <span>Preprocessing</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:20px; height:20px; background:#EEF2FF; border-radius:5px;
                            font-size:10px; font-weight:700; color:#4F46E5; display:flex;
                            align-items:center; justify-content:center;">3</div>
                <span>NLP / Stemming</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:20px; height:20px; background:#EEF2FF; border-radius:5px;
                            font-size:10px; font-weight:700; color:#4F46E5; display:flex;
                            align-items:center; justify-content:center;">4</div>
                <span>Sentiment Transformer</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:20px; height:20px; background:#EEF2FF; border-radius:5px;
                            font-size:10px; font-weight:700; color:#4F46E5; display:flex;
                            align-items:center; justify-content:center;">5</div>
                <span>ML Classification</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown("""
    <div class="hero-wrap">
        <div style="display:inline-block; background:#EEF2FF; color:#4F46E5; font-size:11px;
                    font-weight:700; text-transform:uppercase; letter-spacing:0.1em;
                    padding:4px 12px; border-radius:99px; margin-bottom:14px;">
            Web Mining · Analisis Sentimen
        </div>
        <div class="hero-title">
            SentiTrack <span class="hero-accent">Agro Bromo</span>
        </div>
        <div class="hero-sub">
            Analisis sentimen berbasis transformer terhadap komentar YouTube dari TVONE, KOMPAS, dan METROTV
            mengenai kecelakaan Kereta Api Agro Bromo.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Total Komentar",    "~3.2K",   "3 channel YouTube",          None),
        ("Model Utama",       "RoBERTa", "Indonesian fine-tuned",       None),
        ("Akurasi ML",        "~85%",    "XGBoost + TF-IDF",            "#059669"),
        ("Sentimen Dominan",  "Negatif", "Respons publik",              "#DC2626"),
    ]
    for col, (label, value, sub, color) in zip([col1,col2,col3,col4], cards):
        val_style = f"color:{color};" if color else ""
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:28px; {val_style}">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Channel YouTube</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    channels = [
        ("TVONE",   "#B91C1C", "tag-tvone",   "qHr-ky9Iwik", "Berita kecelakaan & breaking news"),
        ("KOMPAS",  "#1D4ED8", "tag-kompas",  "-lvgdiR6z1g", "Investigasi mendalam & feature"),
        ("METROTV", "#065F46", "tag-metrotv", "5EHTgRAyyMw", "Analisis & diskusi panel"),
    ]
    for col, (name, color, tag_cls, vid_id, desc) in zip([col1,col2,col3], channels):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {color};">
                <span class="channel-tag {tag_cls}">{name}</span>
                <div style="font-family: monospace; font-size:11px; color:#9CA3AF; margin: 10px 0 6px;">{vid_id}</div>
                <div style="font-size:13px; color:#374151; line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Rumusan Masalah</div>', unsafe_allow_html=True)

    rms = [
        ("RM 1", "Distribusi Sentimen",
         "Bagaimana distribusi sentimen netizen terhadap pemberitaan kecelakaan KA Agro Bromo secara keseluruhan?"),
        ("RM 2", "Pola per Channel",
         "Apakah terdapat perbedaan pola sentimen antara channel TVONE, KOMPAS, dan METROTV?"),
        ("RM 3", "Performa ML",
         "Bagaimana performa model ML (Random Forest & XGBoost) dalam mengklasifikasikan sentimen?"),
    ]
    for rm_id, rm_title, rm_desc in rms:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:10px; display:flex; gap:16px; align-items:flex-start;">
            <div style="background:#EEF2FF; color:#4F46E5; border-radius:8px; padding:8px 12px;
                        font-family:'Syne',sans-serif; font-size:13px; font-weight:700;
                        white-space:nowrap; min-width:52px; text-align:center; flex-shrink:0;">
                {rm_id}
            </div>
            <div>
                <div style="font-family:'Syne',sans-serif; font-weight:600; font-size:14px;
                            margin-bottom:4px; color:#111827;">{rm_title}</div>
                <div style="font-size:13px; color:#6B7280; line-height:1.6;">{rm_desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE: Analisis Sentimen
# ══════════════════════════════════════════════════════
elif "Analisis Sentimen" in page:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title" style="font-size:40px;">Analisis <span class="hero-accent">Sentimen</span></div>
        <div class="hero-sub">Input komentar YouTube untuk dianalisis sentimennya secara real-time.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px; color:#9CA3AF; margin-bottom:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.1em;">Contoh Komentar</div>', unsafe_allow_html=True)
    ex_cols = st.columns(3)
    examples = [
        ("Negatif",  "Ini sungguh tragedi yang sangat menyedihkan. Pemerintah harus segera bertindak dan memperbaiki sistem keselamatan kereta api di Indonesia!"),
        ("Netral",   "Saya melihat laporan ini dari Metro TV, tampaknya ada masalah teknis pada sistem pengereman. Perlu investigasi lebih lanjut."),
        ("Positif",  "Terima kasih kepada seluruh tim penyelamat yang sudah bekerja keras membantu korban. Semoga para korban cepat sembuh."),
    ]
    for col, (label, text) in zip(ex_cols, examples):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state['input_text'] = text

    st.markdown('<br>', unsafe_allow_html=True)

    user_input = st.text_area(
        "Komentar:",
        value=st.session_state.get('input_text', ''),
        height=130,
        placeholder="Tulis atau tempel komentar YouTube di sini...",
        label_visibility="collapsed",
        key="main_input"
    )

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        if user_input:
            word_count = len(user_input.split())
            char_count = len(user_input)
            st.markdown(f'<div style="font-size:12px; color:#9CA3AF; margin-top:6px;">{word_count} kata · {char_count} karakter</div>', unsafe_allow_html=True)
    with col_btn:
        analyze_btn = st.button("Analisis →", use_container_width=True)

    if show_preprocessing and user_input:
        with st.expander("Hasil Preprocessing"):
            p1 = clean_text(user_input)
            p2 = normalize_slang(p1)
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Clean Text**")
                st.code(p1, language=None)
            with cols[1]:
                st.markdown("**Normalized (Slang)**")
                st.code(p2, language=None)

    if analyze_btn and user_input.strip():
        with st.spinner("Menganalisis sentimen..."):
            try:
                result = analyze_sentiment(
                    user_input, model_choice,
                    load_roberta() if model_choice == "RoBERTa" else None,
                    load_indobert() if model_choice == "IndoBERT" else None
                )

                label = result['label']
                score = result['score']
                scores = result['scores']

                color_map = {'positive': '#059669', 'neutral': '#D97706', 'negative': '#DC2626'}
                bg_map    = {'positive': '#D1FAE5', 'neutral': '#FEF3C7', 'negative': '#FEE2E2'}
                label_id  = {'positive': 'Positif',  'neutral': 'Netral',   'negative': 'Negatif'}
                color  = color_map.get(label, '#6B7280')
                bg     = bg_map.get(label, '#F3F4F6')

                st.markdown(f"""
                <div class="result-card" style="border-top: 4px solid {color};">
                    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:20px;">
                        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.1em;
                                    color:#9CA3AF; font-weight:700;">Hasil Analisis · {model_choice}</div>
                        <span class="badge badge-{label}">{label_id.get(label, label.capitalize())}</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:20px; margin-bottom:24px;">
                        <div style="width:64px; height:64px; background:{bg}; border-radius:16px;
                                    display:flex; align-items:center; justify-content:center;
                                    font-family:'Syne',sans-serif; font-weight:800; font-size:28px;
                                    color:{color};">{label_id.get(label,'?')[0]}</div>
                        <div>
                            <div style="font-family:'Syne',sans-serif; font-size:32px; font-weight:800;
                                        color:{color}; line-height:1;">{label_id.get(label, label.capitalize())}</div>
                            <div style="font-size:13px; color:#9CA3AF; margin-top:4px;">
                                Konfiden: <strong style="color:#374151;">{score*100:.1f}%</strong>
                            </div>
                        </div>
                    </div>
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.1em;
                                color:#9CA3AF; font-weight:700; margin-bottom:14px;">Skor Per Kelas</div>
                """, unsafe_allow_html=True)

                for sent_key, sent_label in [('positive','Positif'), ('neutral','Netral'), ('negative','Negatif')]:
                    s = scores.get(sent_key, 0.0) * 100
                    c = color_map[sent_key]
                    st.markdown(f"""
                    <div style="margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:5px;">
                            <span style="color:#374151; font-weight:500;">{sent_label}</span>
                            <span style="color:{c}; font-weight:700;">{s:.1f}%</span>
                        </div>
                        <div class="score-bar-container">
                            <div class="score-bar" style="width:{s}%; background:{c};"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}\n\nPastikan model terinstal: `pip install transformers torch`")

    elif analyze_btn:
        st.warning("Silakan masukkan teks terlebih dahulu.")


# ══════════════════════════════════════════════════════
# PAGE: Statistik & EDA
# ══════════════════════════════════════════════════════
elif "Statistik" in page:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title" style="font-size:40px;">Statistik <span class="hero-accent">&amp; EDA</span></div>
        <div class="hero-sub">Eksplorasi distribusi dan pola data komentar YouTube.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.info("Halaman ini menampilkan visualisasi berbasis data crawling. Upload file CSV hasil crawling untuk melihat statistik real.")
    uploaded = st.file_uploader("Upload CSV hasil analisis (opsional)", type=["csv"])

    MPLRC = {
        'figure.facecolor': '#FFFFFF', 'axes.facecolor': '#F9FAFB',
        'text.color': '#111827', 'axes.labelcolor': '#6B7280',
        'xtick.color': '#6B7280', 'ytick.color': '#6B7280',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': True,
        'axes.edgecolor': '#E4E6EF', 'grid.color': '#F3F4F6',
        'axes.grid': True, 'grid.alpha': 0.8,
    }
    COLORS = {'positive':'#059669','neutral':'#D97706','negative':'#DC2626'}

    if uploaded:
        import io
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        st.success(f"Dataset loaded: {len(df):,} baris, {len(df.columns)} kolom")

        if 'sentiment' in df.columns:
            st.markdown('<div class="section-title">Distribusi Sentimen</div>', unsafe_allow_html=True)
            try:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.rcParams.update(MPLRC)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                fig.patch.set_facecolor('#FFFFFF')

                sent_counts = df['sentiment'].value_counts()
                sent_order = [s for s in ['positive','neutral','negative'] if s in sent_counts.index]
                colors = [COLORS.get(s,'#9CA3AF') for s in sent_order]

                axes[0].bar(sent_order, [sent_counts.get(s,0) for s in sent_order],
                            color=colors, edgecolor='none', width=0.5)
                axes[0].set_title('Distribusi Sentimen', fontsize=13, pad=12, fontweight='bold')
                axes[0].set_ylabel('Jumlah', color='#6B7280')

                if 'Source' in df.columns:
                    ct = df.groupby(['Source','sentiment']).size().unstack(fill_value=0)
                    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
                    bottom = np.zeros(len(ct_pct))
                    for sent in sent_order:
                        if sent in ct_pct.columns:
                            axes[1].bar(ct_pct.index, ct_pct[sent].values,
                                        bottom=bottom, label=sent.capitalize(),
                                        color=COLORS.get(sent,'#9CA3AF'), alpha=0.9, edgecolor='none')
                            bottom += ct_pct[sent].values
                    axes[1].set_title('Sentimen per Channel (%)', fontsize=13, pad=12, fontweight='bold')
                    axes[1].legend(fontsize=9, facecolor='#FFFFFF', edgecolor='#E4E6EF')

                plt.tight_layout()
                st.pyplot(fig)
            except ImportError:
                st.warning("Install matplotlib: `pip install matplotlib`")

        st.markdown('<div class="section-title">Preview Data</div>', unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True, height=300)

    else:
        st.markdown('<div class="section-title">Demo Visualisasi (Data Sampel)</div>', unsafe_allow_html=True)
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams.update(MPLRC)

            np.random.seed(42)
            n = 3200
            channels   = np.random.choice(['TVONE','KOMPAS','METROTV'], n, p=[0.35,0.3,0.35])
            sentiments = np.random.choice(['positive','neutral','negative'], n, p=[0.22,0.31,0.47])
            df_demo    = pd.DataFrame({'Source': channels, 'sentiment': sentiments,
                                       'word_count': np.random.poisson(15, n)})

            CHANNEL_COLORS = {'TVONE':'#B91C1C','KOMPAS':'#1D4ED8','METROTV':'#065F46'}

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.patch.set_facecolor('#FFFFFF')

            sent_counts = df_demo['sentiment'].value_counts().reindex(['positive','neutral','negative'], fill_value=0)
            bars = axes[0].bar(sent_counts.index, sent_counts.values,
                               color=[COLORS[s] for s in sent_counts.index],
                               edgecolor='none', width=0.5)
            for bar, val in zip(bars, sent_counts.values):
                pct = val / n * 100
                axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                             f'{pct:.1f}%', ha='center', fontsize=10,
                             fontweight='bold', color='#374151')
            axes[0].set_title('Distribusi Sentimen', fontsize=13, pad=12, fontweight='bold')

            ct = df_demo.groupby(['Source','sentiment']).size().unstack(fill_value=0)
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            bottom = np.zeros(len(ct_pct))
            for sent in ['positive','neutral','negative']:
                if sent in ct_pct.columns:
                    axes[1].bar(ct_pct.index, ct_pct[sent].values,
                                bottom=bottom, label=sent.capitalize(),
                                color=COLORS[sent], alpha=0.9, edgecolor='none')
                    for i,(v,b) in enumerate(zip(ct_pct[sent].values, bottom)):
                        if v > 6:
                            axes[1].text(i, b+v/2, f'{v:.0f}%', ha='center', va='center',
                                         fontsize=9, fontweight='bold', color='#fff')
                    bottom += ct_pct[sent].values
            axes[1].set_title('Sentimen per Channel (%)', fontsize=13, pad=12, fontweight='bold')
            axes[1].legend(fontsize=9, facecolor='#FFFFFF', edgecolor='#E4E6EF')

            for src, color in CHANNEL_COLORS.items():
                data = df_demo[df_demo['Source']==src]['word_count']
                axes[2].hist(data, bins=20, alpha=0.65, label=src, color=color, edgecolor='none')
            axes[2].set_title('Distribusi Jumlah Kata', fontsize=13, pad=12, fontweight='bold')
            axes[2].set_xlabel('Jumlah Kata', color='#6B7280')
            axes[2].legend(fontsize=9, facecolor='#FFFFFF', edgecolor='#E4E6EF')

            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Data simulasi untuk demo. Upload CSV hasil crawling untuk data asli.")

        except ImportError:
            st.warning("Install matplotlib: `pip install matplotlib`")


# ══════════════════════════════════════════════════════
# PAGE: Batch Prediksi
# ══════════════════════════════════════════════════════
elif "Batch" in page:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title" style="font-size:40px;">Batch <span class="hero-accent">Prediksi</span></div>
        <div class="hero-sub">Analisis sentimen untuk banyak komentar sekaligus.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Input Manual", "Upload CSV"])

    with tab1:
        bulk_text = st.text_area(
            "Komentar:",
            height=200,
            placeholder="Komentar pertama...\nKomentar kedua...\nKomentar ketiga...",
            label_visibility="collapsed"
        )

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            max_batch = st.slider("Maks komentar diproses", 5, 50, 10)
        with col_opt2:
            st.markdown('<br>', unsafe_allow_html=True)
            batch_btn = st.button("Jalankan Batch Analisis", use_container_width=True)

        if batch_btn and bulk_text.strip():
            lines = [l.strip() for l in bulk_text.strip().split('\n') if l.strip()][:max_batch]
            st.markdown(f'<div style="font-size:13px; color:#9CA3AF; margin: 16px 0 8px;">Memproses {len(lines)} komentar...</div>', unsafe_allow_html=True)

            progress_bar = st.progress(0)
            results = []

            try:
                clf_r = load_roberta() if model_choice == "RoBERTa" else None
                clf_i = load_indobert() if model_choice == "IndoBERT" else None

                for i, line in enumerate(lines):
                    res = analyze_sentiment(line, model_choice, clf_r, clf_i)
                    results.append({
                        'Komentar':  line[:80] + ('...' if len(line)>80 else ''),
                        'Sentimen':  res['label'].capitalize(),
                        'Konfiden':  f"{res['score']*100:.1f}%",
                        'Positif':   f"{res['scores'].get('positive',0)*100:.1f}%",
                        'Netral':    f"{res['scores'].get('neutral',0)*100:.1f}%",
                        'Negatif':   f"{res['scores'].get('negative',0)*100:.1f}%",
                    })
                    progress_bar.progress((i+1)/len(lines))

                df_results = pd.DataFrame(results)
                st.markdown('<div class="section-title">Hasil</div>', unsafe_allow_html=True)
                st.dataframe(df_results, use_container_width=True)

                sent_summary = df_results['Sentimen'].value_counts()
                cols = st.columns(3)
                color_map = {'Positive':'#059669','Neutral':'#D97706','Negative':'#DC2626',
                             'Positif':'#059669','Netral':'#D97706','Negatif':'#DC2626'}
                bg_map    = {'Positive':'#D1FAE5','Neutral':'#FEF3C7','Negative':'#FEE2E2',
                             'Positif':'#D1FAE5','Netral':'#FEF3C7','Negatif':'#FEE2E2'}
                for i, (label, count) in enumerate(sent_summary.items()):
                    pct   = count / len(df_results) * 100
                    color = color_map.get(label,'#6B7280')
                    bg    = bg_map.get(label,'#F3F4F6')
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align:center;">
                            <div style="width:40px; height:40px; background:{bg}; border-radius:10px;
                                        margin: 0 auto 10px; display:flex; align-items:center; justify-content:center;
                                        font-family:'Syne',sans-serif; font-weight:800; font-size:18px; color:{color};">
                                {label[0]}
                            </div>
                            <div style="font-family:'Syne',sans-serif; font-weight:700; color:{color};
                                        font-size:15px;">{label}</div>
                            <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800;
                                        color:#111827;">{count}</div>
                            <div style="color:#9CA3AF; font-size:12px;">{pct:.1f}%</div>
                        </div>""", unsafe_allow_html=True)

                csv_out = df_results.to_csv(index=False)
                st.download_button("Download Hasil CSV", csv_out,
                                   file_name="batch_sentiment_results.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        uploaded_csv = st.file_uploader("Upload CSV dengan kolom 'Comment' atau 'comment'", type=["csv"])
        if uploaded_csv:
            import io
            df_up = pd.read_csv(io.BytesIO(uploaded_csv.getvalue()))
            comment_col = next(
                (c for c in df_up.columns if c.lower() in ['comment','komentar','text','teks']), None
            )

            if comment_col:
                st.success(f"{len(df_up):,} baris ditemukan. Kolom teks: '{comment_col}'")
                n_process = st.slider("Jumlah baris diproses", 10, min(500, len(df_up)), min(50, len(df_up)))

                if st.button("Jalankan pada CSV", use_container_width=True):
                    sample = df_up[comment_col].dropna().head(n_process).tolist()
                    prog = st.progress(0)

                    try:
                        clf_r = load_roberta() if model_choice == "RoBERTa" else None
                        clf_i = load_indobert() if model_choice == "IndoBERT" else None
                        labels = []
                        for idx, text in enumerate(sample):
                            r = analyze_sentiment(str(text), model_choice, clf_r, clf_i)
                            labels.append(r['label'])
                            prog.progress((idx+1)/len(sample))

                        df_out = df_up.head(n_process).copy()
                        df_out['predicted_sentiment'] = labels
                        st.dataframe(df_out, use_container_width=True, height=300)
                        csv_dl = df_out.to_csv(index=False)
                        st.download_button("Download CSV", csv_dl,
                                           file_name="predicted_sentiments.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Kolom 'Comment', 'comment', 'text', atau 'teks' tidak ditemukan. Pastikan CSV memiliki header yang sesuai.")


# ══════════════════════════════════════════════════════
# PAGE: Tentang
# ══════════════════════════════════════════════════════
elif "Tentang" in page:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title" style="font-size:40px;">Tentang <span class="hero-accent">Proyek</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card" style="margin-bottom: 14px;">
        <div style="font-family:'Syne',sans-serif; font-size:15px; font-weight:700;
                    margin-bottom:10px; color:#4F46E5;">Deskripsi</div>
        <div style="font-size:14px; color:#374151; line-height:1.8;">
            Proyek ini merupakan implementasi <strong>Web Mining dan Analisis Sentimen</strong> terhadap komentar YouTube
            pada video pemberitaan kecelakaan Kereta Api Agro Bromo. Data dikumpulkan dari tiga channel berita besar
            Indonesia: <strong style="color:#B91C1C;">TVONE</strong>, <strong style="color:#1D4ED8;">KOMPAS TV</strong>,
            dan <strong style="color:#065F46;">Metro TV</strong>.
        </div>
    </div>

    <div class="metric-card" style="margin-bottom: 14px;">
        <div style="font-family:'Syne',sans-serif; font-size:15px; font-weight:700;
                    margin-bottom:10px; color:#4F46E5;">Teknologi</div>
        <div style="font-size:14px; color:#374151; line-height:2.1;">
            <strong>Data Collection:</strong> YouTube Data API v3<br>
            <strong>Preprocessing:</strong> Sastrawi Stemmer, NLTK, Custom Slang Dict (~200+ kata)<br>
            <strong>Sentiment Labeling:</strong> IndoBERT & RoBERTa (Indonesian fine-tuned)<br>
            <strong>ML Classification:</strong> TF-IDF + Random Forest + XGBoost<br>
            <strong>Topic Modelling:</strong> LDA (Gensim)<br>
            <strong>Web App:</strong> Streamlit
        </div>
    </div>

    <div class="metric-card" style="margin-bottom: 14px;">
        <div style="font-family:'Syne',sans-serif; font-size:15px; font-weight:700;
                    margin-bottom:10px; color:#4F46E5;">Pipeline</div>
        <div style="display:flex; flex-wrap:wrap; gap:8px; font-size:13px;">
            <span style="background:#EEF2FF; color:#4F46E5; padding:6px 14px; border-radius:8px; font-weight:600;">1. Install & Import</span>
            <span style="background:#EEF2FF; color:#4F46E5; padding:6px 14px; border-radius:8px; font-weight:600;">2. Web Crawling</span>
            <span style="background:#EEF2FF; color:#4F46E5; padding:6px 14px; border-radius:8px; font-weight:600;">3. Preprocessing</span>
            <span style="background:#EEF2FF; color:#4F46E5; padding:6px 14px; border-radius:8px; font-weight:600;">4. EDA</span>
            <span style="background:#EEF2FF; color:#4F46E5; padding:6px 14px; border-radius:8px; font-weight:600;">5. Sentiment Analysis</span>
            <span style="background:#EEF2FF; color:#4F46E5; padding:6px 14px; border-radius:8px; font-weight:600;">6. Web Mining</span>
            <span style="background:#EEF2FF; color:#4F46E5; padding:6px 14px; border-radius:8px; font-weight:600;">7. ML Classification</span>
            <span style="background:#EEF2FF; color:#4F46E5; padding:6px 14px; border-radius:8px; font-weight:600;">8. Insight</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Cara Instalasi</div>', unsafe_allow_html=True)
    st.code("""# 1. Install dependencies
pip install streamlit transformers torch sastrawi nltk
pip install scikit-learn xgboost gensim wordcloud
pip install google-api-python-client matplotlib seaborn

# 2. Jalankan
streamlit run app.py
""", language="bash")

    st.markdown('<div class="section-title">Catatan Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <div style="font-size:13px; color:#374151; line-height:2;">
            <strong>RoBERTa:</strong> <code>w11wo/indonesian-roberta-base-sentiment-classifier</code><br>
            <strong>IndoBERT:</strong> <code>mdhugol/indonesia-bert-sentiment-classification</code><br>
            Model akan otomatis diunduh dari Hugging Face saat pertama dijalankan.
        </div>
    </div>
    """, unsafe_allow_html=True)
