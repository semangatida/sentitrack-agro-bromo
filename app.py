import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# ── Page Config 
st.set_page_config(
    page_title="SentiTrack | Agro Bromo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS 
st.markdown("""
<style>
:root {
    --bg: #FFFFFF;
    --surface: #F8F9FA;
    --border: #E5E7EB;
    --accent: #111827;
    --positive: #059669;
    --neutral: #6B7280;
    --negative: #DC2626;
    --text-main: #111827;
    --text-muted: #6B7280;
}

html, body, [class*="css"] {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    background-color: var(--bg);
    color: var(--text-main);
}

.stApp {
    background-color: var(--bg);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text-main) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Typography */
.hero-title {
    font-size: 42px;
    font-weight: 700;
    line-height: 1.2;
    letter-spacing: -0.02em;
    color: var(--text-main);
    margin-bottom: 8px;
}
.hero-sub {
    font-size: 16px;
    color: var(--text-muted);
    max-width: 600px;
    line-height: 1.6;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-main);
    margin: 32px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* Metric Cards */
.metric-card {
    background-color: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
    transition: box-shadow 0.2s ease;
}
.metric-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}
.metric-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-main);
    line-height: 1;
}
.metric-sub {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 8px;
}

/* Channel Tags */
.channel-tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.tag-tvone   { background-color: #FEE2E2; color: #991B1B; }
.tag-kompas  { background-color: #DBEAFE; color: #1E40AF; }
.tag-metrotv { background-color: #D1FAE5; color: #065F46; }

/* Buttons & Inputs */
.stTextArea textarea, .stSelectbox > div > div {
    background-color: var(--bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    border-radius: 6px !important;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
}
.stTextArea textarea:focus, .stSelectbox > div > div:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

.stButton > button {
    background-color: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 24px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.9;
}

/* Result Card */
.result-card {
    background-color: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
    margin-top: 24px;
}
.score-bar-container {
    background-color: #F3F4F6;
    border-radius: 4px;
    height: 8px;
    margin: 8px 0 16px;
    overflow: hidden;
}
.score-bar {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    font-weight: 600 !important;
}

hr.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 32px 0;
}
</style>
""", unsafe_allow_html=True)


# ── NLP Setup 
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
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=512,
        device=-1,
    )
    return clf

@st.cache_resource(show_spinner=False)
def load_indobert():
    from transformers import pipeline as hf_pipeline
    model_name = "mdhugol/indonesia-bert-sentiment-classification"
    clf = hf_pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=512,
        device=-1,
    )
    return clf


# ── Constants 
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


# ── Text Processing
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

        else:  # IndoBERT
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
        return {'label': 'neutral', 'score': 1.0, 'scores': {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}, 'error': str(e)}


# ── Sidebar 
with st.sidebar:
    st.markdown("""
    <div style="padding: 16px 0 32px;">
        <div style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:var(--text-muted); margin-bottom:8px;">Project Overview</div>
        <div style="font-size:22px; font-weight:700; color:var(--text-main); line-height:1.2;">
            Web Mining &<br>Analisis Sentimen
        </div>
        <div style="font-size:14px; color:var(--text-muted); margin-top:8px;">Kecelakaan KA Agro Bromo</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigasi Modul",
        ["Dashboard Utama", "Analisis Sentimen", "Statistik & Data", "Batch Prediksi", "Tentang Proyek"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border:none; border-top:1px solid var(--border); margin: 32px 0;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:var(--text-muted); margin-bottom:12px;">Pengaturan Model</div>', unsafe_allow_html=True)

    model_choice = st.selectbox("Pilih Model Transformer", ["RoBERTa", "IndoBERT"], key="model_sel")
    show_preprocessing = st.toggle("Tampilkan Log Preprocessing", value=False)

    st.markdown("<hr style='border:none; border-top:1px solid var(--border); margin: 32px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:13px; color:var(--text-muted); line-height:2;">
        <div style="font-weight:700; margin-bottom:8px; font-size:12px; text-transform:uppercase; letter-spacing:0.1em;">Alur Proses</div>
        1. Pengumpulan Data Web<br>
        2. Preprocessing Teks<br>
        3. Ekstraksi Fitur (NLP)<br>
        4. Klasifikasi Transformer<br>
        5. Visualisasi Sentimen
    </div>
    """, unsafe_allow_html=True)


# PAGE: Dashboard

if "Dashboard" in page:
    st.markdown("""
    <div style="padding: 24px 0;">
        <div class="hero-title">SentiTrack Data Overview</div>
        <div class="hero-sub">
            Dashboard analisis sentimen berbasis arsitektur Transformer terhadap respons publik dari saluran berita terkemuka mengenai insiden Kereta Api Agro Bromo.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Sumber Data</div>
            <div class="metric-value">3</div>
            <div class="metric-sub">Channel Media Nasional</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Arsitektur Model</div>
            <div class="metric-value" style="font-size:24px; padding-top:4px;">RoBERTa</div>
            <div class="metric-sub">Indonesian fine-tuned</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Akurasi Dasar</div>
            <div class="metric-value" style="color:#059669;">85.2%</div>
            <div class="metric-sub">Evaluasi Validation Set</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Sentimen Dominan</div>
            <div class="metric-value" style="color:#DC2626;">Negatif</div>
            <div class="metric-sub">Mayoritas respons audiens</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Sumber Saluran Media</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    channels = [
        ("TVONE", "#991B1B", "tag-tvone", "Fokus pemberitaan insiden dan berita terkini."),
        ("KOMPAS", "#1E40AF", "tag-kompas", "Investigasi mendalam dan ulasan kronologi."),
        ("METROTV", "#065F46", "tag-metrotv", "Analisis ahli dan diskusi panel operasional."),
    ]
    for col, (name, color, tag_cls, desc) in zip([col1,col2,col3], channels):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {color}; border-radius: 4px;">
                <span class="channel-tag {tag_cls}">{name}</span>
                <div style="font-size:14px; color:var(--text-muted); margin-top:12px; line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Fokus Analisis</div>', unsafe_allow_html=True)

    rms = [
        ("01", "Distribusi Sentimen Publik", "Bagaimana proporsi sentimen positif, netral, dan negatif dari komentar netizen terhadap penanganan insiden?"),
        ("02", "Pola Karakteristik Media", "Apakah terdapat perbedaan kecenderungan sentimen audiens berdasarkan gaya peliputan masing-masing media?"),
        ("03", "Evaluasi Performa Model", "Bagaimana perbandingan efektivitas model klasifikasi dalam memahami konteks dan slang bahasa Indonesia?"),
    ]
    for rm_id, rm_title, rm_desc in rms:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:16px; display:flex; gap:20px; align-items:flex-start;">
            <div style="color:var(--text-muted); font-size:18px; font-weight:700; padding-top:2px;">
                {rm_id}
            </div>
            <div>
                <div style="font-weight:700; font-size:16px; margin-bottom:6px; color:var(--text-main);">{rm_title}</div>
                <div style="font-size:14px; color:var(--text-muted); line-height:1.6;">{rm_desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)


# PAGE: Analisis Sentimen (Single Input)

elif "Analisis Sentimen" in page:
    st.markdown("""
    <div style="padding: 24px 0;">
        <div class="hero-title">Pengujian Sentimen</div>
        <div class="hero-sub">Lakukan klasifikasi sentimen pada teks secara real-time menggunakan model NLP yang telah dilatih.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div style="font-size:12px; font-weight:600; color:var(--text-muted); margin-bottom:12px; text-transform:uppercase;">Pilih Teks Sampel</div>', unsafe_allow_html=True)
    ex_cols = st.columns(3)
    examples = [
        ("Konteks Negatif", "Sistem keamanan kereta harus segera dievaluasi total, kejadian ini sangat mengecewakan."),
        ("Konteks Netral", "Menurut informasi terbaru, pihak KNKT sedang melakukan proses investigasi di lapangan."),
        ("Konteks Positif", "Apresiasi untuk tim SAR dan petugas medis yang sigap melakukan evakuasi korban."),
    ]
    for col, (label, text) in zip(ex_cols, examples):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state['input_text'] = text

    st.markdown('<br>', unsafe_allow_html=True)

    user_input = st.text_area(
        "Teks Input:",
        value=st.session_state.get('input_text', ''),
        height=140,
        placeholder="Masukkan teks ulasan atau komentar di sini...",
        label_visibility="collapsed",
        key="main_input"
    )

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        if user_input:
            word_count = len(user_input.split())
            char_count = len(user_input)
            st.markdown(f'<div style="font-size:13px; color:var(--text-muted); margin-top:8px; font-weight:500;">Detail Input: {word_count} kata | {char_count} karakter</div>', unsafe_allow_html=True)
    with col_btn:
        analyze_btn = st.button("Proses Teks", use_container_width=True)

    if show_preprocessing and user_input:
        with st.expander("Inspeksi Pra-pemrosesan Teks"):
            p1 = clean_text(user_input)
            p2 = normalize_slang(p1)
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Tahap 1: Pembersihan Karakter**")
                st.code(p1, language=None)
            with cols[1]:
                st.markdown("**Tahap 2: Normalisasi Slang**")
                st.code(p2, language=None)

    if analyze_btn and user_input.strip():
        with st.spinner("Memproses analisis teks..."):
            try:
                if model_choice == "RoBERTa":
                    clf = load_roberta()
                else:
                    clf = load_indobert()

                result = analyze_sentiment(
                    user_input, model_choice,
                    load_roberta() if model_choice == "RoBERTa" else None,
                    load_indobert() if model_choice == "IndoBERT" else None
                )

                label = result['label']
                score = result['score']
                scores = result['scores']

                color_map = {'positive': '#059669', 'neutral': '#6B7280', 'negative': '#DC2626'}
                label_id  = {'positive': 'Positif', 'neutral': 'Netral', 'negative': 'Negatif'}
                color = color_map.get(label, '#111827')

                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid {color};">
                    <div style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:var(--text-muted);">Kesimpulan Model ({model_choice})</div>
                    <div style="margin: 16px 0 24px;">
                        <div style="font-size:32px; font-weight:700; color:{color}; line-height:1;">
                            {label_id.get(label, label.capitalize())}
                        </div>
                        <div style="font-size:14px; color:var(--text-muted); margin-top:8px; font-weight:500;">Tingkat Kepercayaan: {score*100:.1f}%</div>
                    </div>
                    <div style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:var(--text-muted); margin-bottom:16px;">Distribusi Probabilitas Kelas</div>
                """, unsafe_allow_html=True)

                for sent_key, sent_label in [('positive','Positif'), ('neutral','Netral'), ('negative','Negatif')]:
                    s = scores.get(sent_key, 0.0) * 100
                    c = color_map[sent_key]
                    st.markdown(f"""
                    <div style="margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; font-size:14px; font-weight:500; margin-bottom:6px;">
                            <span style="color:var(--text-main);">{sent_label}</span>
                            <span style="color:{c}; font-weight:700;">{s:.1f}%</span>
                        </div>
                        <div class="score-bar-container">
                            <div class="score-bar" style="width:{s}%; background-color:{c};"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan sistem: {e}")

    elif analyze_btn:
        st.warning("Mohon masukkan teks sebelum melakukan pemrosesan.")



# PAGE: Statistik & Data

elif "Statistik" in page:
    st.markdown("""
    <div style="padding: 24px 0;">
        <div class="hero-title">Eksplorasi Data Base</div>
        <div class="hero-sub">Visualisasi statistik deskriptif dari dataset hasil penarikan data (crawling).</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.info("Halaman ini dikhususkan untuk menampilkan data historis. Silakan unggah file CSV hasil pengumpulan data untuk melihat distribusi.")

    uploaded = st.file_uploader("Unggah dataset CSV", type=["csv"])

    if uploaded:
        import io
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        st.success(f"Berhasil memuat {len(df):,} baris data.")

        if 'sentiment' in df.columns and 'Source' in df.columns:
            st.markdown('<div class="section-title">Visualisasi Distribusi Sentimen</div>', unsafe_allow_html=True)

            try:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.rcParams.update({
                    'font.family': 'sans-serif',
                    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial'],
                    'figure.facecolor': '#FFFFFF',
                    'axes.facecolor': '#FFFFFF',
                    'text.color': '#111827',
                    'axes.labelcolor': '#6B7280',
                    'xtick.color': '#6B7280',
                    'ytick.color': '#6B7280',
                    'axes.spines.top': False,
                    'axes.spines.right': False,
                    'axes.spines.left': False,
                    'axes.edgecolor': '#E5E7EB',
                    'grid.color': '#F3F4F6',
                    'axes.grid': True,
                    'grid.axis': 'y'
                })

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                COLORS = {'positive':'#059669','neutral':'#6B7280','negative':'#DC2626'}
                
                sent_counts = df['sentiment'].value_counts()
                sent_order = [s for s in ['positive','neutral','negative'] if s in sent_counts.index]
                colors = [COLORS.get(s,'#111827') for s in sent_order]

                axes[0].bar([s.capitalize() for s in sent_order], [sent_counts.get(s,0) for s in sent_order],
                            color=colors, width=0.5)
                axes[0].set_title('Frekuensi Keseluruhan', fontsize=14, fontweight='bold', pad=16)
                axes[0].set_ylabel('Jumlah Komentar')

                if 'Source' in df.columns:
                    ct = df.groupby(['Source','sentiment']).size().unstack(fill_value=0)
                    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
                    bottom = np.zeros(len(ct_pct))
                    
                    for sent in sent_order:
                        if sent in ct_pct.columns:
                            axes[1].bar(ct_pct.index, ct_pct[sent].values,
                                        bottom=bottom, label=sent.capitalize(),
                                        color=COLORS.get(sent,'#111827'))
                            bottom += ct_pct[sent].values
                            
                    axes[1].set_title('Komposisi Per Saluran Media (%)', fontsize=14, fontweight='bold', pad=16)
                    axes[1].legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')

                plt.tight_layout()
                st.pyplot(fig)
            except ImportError:
                st.warning("Pustaka Matplotlib diperlukan untuk merender visualisasi.")

        st.markdown('<div class="section-title">Pratinjau Tabel Data</div>', unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True, height=400)

    else:
        st.markdown('<div class="section-title">Data Simulasi (Contoh Tampilan)</div>', unsafe_allow_html=True)
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial'],
                'figure.facecolor': '#FFFFFF',
                'axes.facecolor': '#FFFFFF',
                'text.color': '#111827',
                'axes.labelcolor': '#6B7280',
                'xtick.color': '#6B7280',
                'ytick.color': '#6B7280',
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.spines.left': False,
                'axes.edgecolor': '#E5E7EB',
                'grid.color': '#F3F4F6',
                'axes.grid': True,
                'grid.axis': 'y'
            })

            np.random.seed(42)
            n = 3200
            channels = np.random.choice(['TVONE','KOMPAS','METROTV'], n, p=[0.35,0.3,0.35])
            sentiments = np.random.choice(['positive','neutral','negative'], n, p=[0.22,0.31,0.47])
            df_demo = pd.DataFrame({'Source': channels, 'sentiment': sentiments})

            COLORS = {'positive':'#059669','neutral':'#6B7280','negative':'#DC2626'}

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            sent_counts = df_demo['sentiment'].value_counts().reindex(['positive','neutral','negative'])
            axes[0].bar([s.capitalize() for s in sent_counts.index], sent_counts.values,
                        color=[COLORS[s] for s in sent_counts.index], width=0.5)
            axes[0].set_title('Frekuensi Keseluruhan', fontsize=14, fontweight='bold', pad=16)

            ct = df_demo.groupby(['Source','sentiment']).size().unstack(fill_value=0)
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            bottom = np.zeros(len(ct_pct))
            
            for sent in ['positive','neutral','negative']:
                axes[1].bar(ct_pct.index, ct_pct[sent].values, bottom=bottom, 
                            label=sent.capitalize(), color=COLORS[sent])
                bottom += ct_pct[sent].values
                
            axes[1].set_title('Komposisi Per Saluran Media (%)', fontsize=14, fontweight='bold', pad=16)
            axes[1].legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            st.pyplot(fig)
        except ImportError:
            pass



# PAGE: Batch Prediksi

elif "Batch" in page:
    st.markdown("""
    <div style="padding: 24px 0;">
        <div class="hero-title">Pemrosesan Batch</div>
        <div class="hero-sub">Lakukan klasifikasi sentimen dalam jumlah besar menggunakan skema input baris atau unggah berkas CSV.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Input Manual Multiple", "Unggah Berkas Data"])

    with tab1:
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        bulk_text = st.text_area(
            "Masukkan beberapa baris teks:",
            height=200,
            placeholder="Baris teks pertama...\nBaris teks kedua...\nBaris teks ketiga...",
            label_visibility="collapsed"
        )

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            max_batch = st.slider("Batas maksimal pemrosesan", 5, 50, 10)
        with col_opt2:
            st.markdown('<br>', unsafe_allow_html=True)
            batch_btn = st.button("Jalankan Pemrosesan Manual", use_container_width=True)

        if batch_btn and bulk_text.strip():
            lines = [l.strip() for l in bulk_text.strip().split('\n') if l.strip()][:max_batch]
            st.markdown(f'<div style="font-size:14px; color:var(--text-muted); margin: 16px 0 8px; font-weight:500;">Sistem sedang memproses {len(lines)} entri data...</div>', unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            results = []

            try:
                clf_r = load_roberta() if model_choice == "RoBERTa" else None
                clf_i = load_indobert() if model_choice == "IndoBERT" else None

                for i, line in enumerate(lines):
                    res = analyze_sentiment(line, model_choice, clf_r, clf_i)
                    results.append({
                        'Teks Input': line[:100] + ('...' if len(line)>100 else ''),
                        'Label Sentimen': res['label'].capitalize(),
                        'Probabilitas Output': f"{res['score']*100:.1f}%"
                    })
                    progress_bar.progress((i+1)/len(lines))

                df_results = pd.DataFrame(results)
                st.markdown('<div class="section-title">Ringkasan Hasil Output</div>', unsafe_allow_html=True)
                st.dataframe(df_results, use_container_width=True)

                csv_out = df_results.to_csv(index=False)
                st.download_button("Unduh Ekspor Data", csv_out, file_name="output_batch_manual.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Sistem gagal mengeksekusi perintah: {e}")

    with tab2:
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        uploaded_csv = st.file_uploader("Unggah CSV (Sistem mendeteksi kolom 'Comment', 'Komentar', atau 'Text')", type=["csv"])
        
        if uploaded_csv:
            import io
            df_up = pd.read_csv(io.BytesIO(uploaded_csv.getvalue()))
            comment_col = next((c for c in df_up.columns if c.lower() in ['comment','komentar','text','teks']), None)

            if comment_col:
                st.success(f"Kolom target terdeteksi: '{comment_col}'. Total baris tersedia: {len(df_up):,}")
                n_process = st.slider("Alokasi pemrosesan baris", 10, min(500, len(df_up)), min(50, len(df_up)))

                if st.button("Mulai Pemrosesan Data Berkala", use_container_width=True):
                    sample = df_up[comment_col].dropna().head(n_process).tolist()
                    prog = st.progress(0)

                    try:
                        clf_r = load_roberta() if model_choice == "RoBERTa" else None
                        clf_i = load_indobert() if model_choice == "IndoBERT" else None
                        labels = []
                        
                        for idx, text in enumerate(sample):
                            r = analyze_sentiment(str(text), model_choice, clf_r, clf_i)
                            labels.append(r['label'].capitalize())
                            prog.progress((idx+1)/len(sample))

                        df_out = df_up.head(n_process).copy()
                        df_out['Hasil_Sentimen'] = labels
                        
                        st.markdown('<div class="section-title">Hasil Prediksi Tabel</div>', unsafe_allow_html=True)
                        st.dataframe(df_out, use_container_width=True, height=400)
                        
                        csv_dl = df_out.to_csv(index=False)
                        st.download_button("Unduh Hasil Anotasi Data", csv_dl, file_name="anotasi_sentimen.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Terjadi kegagalan fungsi iterasi: {e}")
            else:
                st.error("Header kolom yang valid ('Comment', 'Komentar', 'Text') tidak dapat dideteksi dalam skema CSV yang diunggah.")


# PAGE: Tentang
elif "Tentang" in page:
    st.markdown("""
    <div style="padding: 24px 0;">
        <div class="hero-title">Spesifikasi Proyek Sistem</div>
        <div class="hero-sub">Dokumentasi fungsional dan kapabilitas dari infrastruktur analisis yang digunakan.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card" style="margin-bottom: 24px;">
        <div style="font-size:16px; font-weight:700; color:var(--text-main); margin-bottom:12px;">Ruang Lingkup Fungsional</div>
        <div style="font-size:15px; color:var(--text-muted); line-height:1.8;">
            Aplikasi ini mendemonstrasikan implementasi rekayasa data tingkat lanjut (Web Mining) 
            dan klasifikasi sentimen pada teks berbahasa Indonesia dengan memanfaatkan 
            arsitektur transformer modern (RoBERTa & IndoBERT). Topik pengujian berfokus pada analisis 
            opini publik pasca insiden transportasi, dengan mengambil rujukan metrik audiens pada stasiun berita nasional.
        </div>
    </div>

    <div class="metric-card" style="margin-bottom: 24px;">
        <div style="font-size:16px; font-weight:700; color:var(--text-main); margin-bottom:12px;">Spesifikasi Teknologi Utama</div>
        <div style="font-size:14px; color:var(--text-muted); line-height:2.2;">
            <strong style="color:var(--text-main);">Data:</strong> YouTube Data API v3<br>
            <strong style="color:var(--text-main);">Pemrosesan Teks:</strong> Sastrawi Stemmer, NLTK, Custom Mapping Regex<br>
            <strong style="color:var(--text-main);">Komputasi Model:</strong> HuggingFace Pipeline (w11wo/roberta, mdhugol/indobert)<br>
            <strong style="color:var(--text-main);">Klasifikasi:</strong> Scikit-learn (TF-IDF, Random Forest), XGBoost<br>
            <strong style="color:var(--text-main);">Infrastruktur:</strong> Streamlit Framework
        </div>
    </div>

    <div class="metric-card" style="margin-bottom: 24px;">
        <div style="font-size:16px; font-weight:700; color:var(--text-main); margin-bottom:12px;">Arsitektur Aliran Kerja (Workflow)</div>
        <div style="font-size:14px; color:var(--text-muted); line-height:2.2;">
            Tahap 1: Ekstraksi Data berbasis API<br>
            Tahap 2: Pre-processing (Pembersihan, Normalisasi Slang, Stemming)<br>
            Tahap 3: Pelabelan Sentimen Otomatis Menggunakan Pre-trained<br>
            Tahap 4: Ekstraksi Fitur (N-gram, LDA Topic Modeling)<br>
            Tahap 5: Deployment
        </div>
    </div>
    """, unsafe_allow_html=True)
