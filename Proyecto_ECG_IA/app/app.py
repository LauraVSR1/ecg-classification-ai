import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
 
# ── Configuración ─────────────────────────────────────────────
st.set_page_config(
    page_title="ECG · Análisis Cardíaco",
    page_icon="🫀",
    layout="centered"
)
 
# ── CSS personalizado ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
 
    /* ── Fondo con degradado suave ── */
    .stApp {
        background: linear-gradient(145deg, #f0eeff 0%, #fce4f3 35%, #e4eeff 70%, #eef6ff 100%);
        min-height: 100vh;
    }
 
    /* ── Header ── */
    .ecg-header {
        text-align: center;
        padding: 2.8rem 0 1.6rem;
    }
    .ecg-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(160,120,255,0.25);
        border-radius: 99px;
        padding: 5px 16px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #8b5cf6;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }
    .ecg-title {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 0.6rem;
        background: linear-gradient(135deg, #7c3aed 0%, #ec4899 55%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .ecg-sub {
        color: #94a3b8;
        font-size: 0.93rem;
        font-weight: 400;
        max-width: 400px;
        margin: 0 auto;
        line-height: 1.65;
    }
 
    /* ── Divider ── */
    .ecg-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139,92,246,0.25), rgba(236,72,153,0.2), transparent);
        margin: 1.6rem 0;
    }
 
    /* ── Upload area ── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.6) !important;
        border: 2px dashed rgba(139,92,246,0.35) !important;
        border-radius: 18px !important;
        transition: all 0.2s ease;
        backdrop-filter: blur(8px);
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(236,72,153,0.5) !important;
        background: rgba(255,255,255,0.8) !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span {
        color: #a78bfa !important;
    }
 
    /* ── Imagen ── */
    [data-testid="stImage"] img {
        border-radius: 16px;
        border: 2px solid rgba(255,255,255,0.9);
        box-shadow: 0 8px 40px rgba(120,80,200,0.12);
    }
 
    /* ── Tarjetas de resultado ── */
    .result-card {
        border-radius: 20px;
        padding: 1.5rem 1.8rem;
        margin: 1.2rem 0 0.5rem;
        position: relative;
        overflow: hidden;
    }
    .result-card::after {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 120px; height: 120px;
        border-radius: 50%;
        opacity: 0.12;
    }
    .card-normal {
        background: linear-gradient(135deg, rgba(236,253,245,0.95) 0%, rgba(209,250,229,0.9) 100%);
        border: 1.5px solid rgba(52,211,153,0.4);
        box-shadow: 0 4px 24px rgba(52,211,153,0.1);
    }
    .card-normal::after { background: #34d399; }
    .card-anormal {
        background: linear-gradient(135deg, rgba(255,241,242,0.95) 0%, rgba(254,226,226,0.9) 100%);
        border: 1.5px solid rgba(251,113,133,0.4);
        box-shadow: 0 4px 24px rgba(251,113,133,0.1);
    }
    .card-anormal::after { background: #fb7185; }
 
    .card-pill {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 3px 12px;
        border-radius: 99px;
        margin-bottom: 0.7rem;
    }
    .pill-normal  { background: rgba(52,211,153,0.15); color: #059669; }
    .pill-anormal { background: rgba(251,113,133,0.15); color: #e11d48; }
 
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .title-normal  { color: #065f46; }
    .title-anormal { color: #9f1239; }
 
    .card-text {
        font-size: 0.88rem;
        line-height: 1.75;
    }
    .text-normal  { color: #047857; }
    .text-anormal { color: #be123c; }
 
    /* ── Métricas ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.7) !important;
        border: 1.5px solid rgba(255,255,255,0.95) !important;
        border-radius: 16px !important;
        padding: 1.1rem 1.3rem !important;
        box-shadow: 0 2px 16px rgba(120,80,200,0.07) !important;
        backdrop-filter: blur(8px);
    }
    [data-testid="stMetricLabel"] {
        color: #a78bfa !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        color: #1e1b4b !important;
        font-weight: 800 !important;
        font-size: 1.7rem !important;
    }
 
    /* ── Progress bar ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #a78bfa, #ec4899, #60a5fa) !important;
        border-radius: 99px !important;
    }
    .stProgress > div > div > div {
        background: rgba(167,139,250,0.15) !important;
        border-radius: 99px !important;
        height: 9px !important;
    }
 
    /* ── Labels ── */
    .score-caption {
        font-size: 0.78rem;
        color: #c4b5fd;
        margin-top: 0.3rem;
        letter-spacing: 0.4px;
    }
    .progress-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #7c3aed;
        margin-bottom: 0.3rem;
    }
 
    /* ── Disclaimer ── */
    .disclaimer {
        background: rgba(255,255,255,0.6);
        border: 1.5px solid rgba(251,191,36,0.35);
        border-radius: 14px;
        padding: 0.95rem 1.2rem;
        font-size: 0.82rem;
        color: #92400e;
        line-height: 1.65;
        margin-top: 1.2rem;
        backdrop-filter: blur(8px);
    }
 
    /* ── Info & expander ── */
    .stAlert {
        background: rgba(255,255,255,0.65) !important;
        border: 1.5px solid rgba(139,92,246,0.25) !important;
        border-radius: 14px !important;
        color: #7c3aed !important;
        backdrop-filter: blur(8px);
    }
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.55) !important;
        border-radius: 12px !important;
        color: #7c3aed !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.45) !important;
        color: #64748b !important;
        font-size: 0.87rem !important;
        border-radius: 0 0 12px 12px !important;
    }
 
    /* ── Texto general ── */
    p, li { color: #64748b; }
    strong { color: #4c1d95; }
</style>
""", unsafe_allow_html=True)
 
# ── Cargar modelo ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("best_ecg_model_v6.keras")
 
model = cargar_modelo()
 
THRESHOLD = 0.68
 
# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="ecg-header">
    <div class="ecg-badge">🫀 &nbsp; IA Cardíaca</div>
    <div class="ecg-title">Clasificador de ECG</div>
    <div class="ecg-sub">Sube una imagen de electrocardiograma y el modelo determinará si es normal o anormal</div>
</div>
""", unsafe_allow_html=True)
 
st.markdown("<hr class='ecg-divider'>", unsafe_allow_html=True)
 
# ── Upload ────────────────────────────────────────────────────
archivo = st.file_uploader(
    "Selecciona una imagen de ECG",
    type=["jpg", "jpeg", "png"],
    help="Formatos soportados: JPG, JPEG, PNG"
)
 
if archivo is not None:
    imagen = Image.open(archivo).convert("RGB")
    st.image(imagen, caption=f"📂 {archivo.name}", use_column_width=True)
 
    st.markdown("<hr class='ecg-divider'>", unsafe_allow_html=True)
 
    # ── Predicción ────────────────────────────────────────────
    with st.spinner("Analizando el ECG..."):
        img_array = np.array(imagen.resize((224, 224)))
        img_array = np.expand_dims(img_array, axis=0).astype("float32")
        prob = float(model.predict(img_array)[0][0])
 
    es_anormal  = prob > THRESHOLD
    confianza   = prob if es_anormal else (1 - prob)
    confianza_p = round(confianza * 100, 1)
 
    # ── Resultado ─────────────────────────────────────────────
    if not es_anormal:
        st.markdown(f"""
        <div class="result-card card-normal">
            <span class="card-pill pill-normal">Resultado</span>
            <div class="card-title title-normal">✅ ECG Normal</div>
            <div class="card-text text-normal">
                El electrocardiograma analizado no muestra señales de irregularidades.
                Tu ritmo cardíaco luce dentro de los parámetros normales.
                Recuerda mantener tus chequeos médicos periódicos para seguir
                cuidando tu salud cardíaca. ¡Sigue así!
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card card-anormal">
            <span class="card-pill pill-anormal">Resultado</span>
            <div class="card-title title-anormal">⚠️ ECG Anormal</div>
            <div class="card-text text-anormal">
                El electrocardiograma muestra patrones que podrían indicar una
                irregularidad cardíaca. Esto <strong>no es un diagnóstico definitivo</strong>,
                pero es importante que consultes con un médico cardiólogo lo antes
                posible para una evaluación profesional. No ignores esta señal.
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    # ── Métricas ──────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Confianza del modelo", value=f"{confianza_p}%")
    with col2:
        st.metric(label="Resultado", value="Normal" if not es_anormal else "Anormal")
 
    # ── Barra de probabilidad ─────────────────────────────────
    st.markdown("<div class='progress-label'>Probabilidad de anomalía</div>", unsafe_allow_html=True)
    st.progress(prob)
    st.markdown(f"<div class='score-caption'>Score: {prob:.4f} &nbsp;|&nbsp; Threshold: {THRESHOLD}</div>", unsafe_allow_html=True)
 
    # ── Disclaimer ────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚕️ <strong>Aviso importante:</strong> Este resultado es orientativo y
        <strong>no reemplaza el diagnóstico de un profesional de la salud</strong>.
        Consulta siempre a tu médico ante cualquier duda sobre tu salud cardíaca.
        Este modelo es una herramienta de apoyo, no un sustituto de la atención médica.
    </div>
    """, unsafe_allow_html=True)
 
else:
    st.info("👆 Sube una imagen de ECG para comenzar el análisis.")
 
    with st.expander("¿Cómo funciona?"):
        st.markdown("""
        1. **Sube** una imagen de electrocardiograma (JPG o PNG)
        2. El modelo analiza los patrones de la imagen
        3. Recibes un resultado indicando si el ECG es **normal** o **anormal**
        4. Se muestra el nivel de confianza del modelo en su predicción
 
        El modelo fue entrenado con imágenes de ECG clasificadas en 4 categorías:
        normal, infarto de miocardio, historial de infarto y latidos anormales.
        """)