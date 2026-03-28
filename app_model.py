"""
🌍 Landmark Explorer - Streamlit App
=====================================
Ejecutar con:
    streamlit run app.py

Requisitos:
    pip install streamlit Pillow torch torchvision numpy
"""

import streamlit as st
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch
import io

# ─── Configuración de página ─────────────────────────────────────
st.set_page_config(
    page_title="Landmark Explorer",
    page_icon="🌍",
    layout="centered",
)

# ─── CSS custom ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Playfair+Display:wght@700&display=swap');

    /* Ocultar menú y footer de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Fondo general */
    .stApp {
        background: linear-gradient(160deg, #0a0a1a 0%, #1a1a2e 40%, #16213e 100%);
    }

    /* Header custom */
    .hero-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e94560, #ff6b6b, #e94560);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-family: 'DM Sans', sans-serif;
        color: #8888a8;
        text-align: center;
        font-size: 1.05rem;
        margin-top: 4px;
        margin-bottom: 30px;
    }

    /* Card container */
    .upload-card {
        background: rgba(22, 33, 62, 0.6);
        border: 1px solid rgba(233, 69, 96, 0.15);
        border-radius: 16px;
        padding: 2rem;
        backdrop-filter: blur(10px);
    }

    /* Resultado individual */
    .result-item {
        background: rgba(10, 10, 26, 0.5);
        border: 1px solid rgba(233, 69, 96, 0.1);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 14px;
        transition: border-color 0.3s ease;
    }
    .result-item:hover {
        border-color: rgba(233, 69, 96, 0.4);
    }

    .result-rank {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        color: #e94560;
        min-width: 30px;
        text-align: center;
    }

    .result-info {
        flex: 1;
    }

    .result-name {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        color: #e0e0e8;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }

    .bar-track {
        width: 100%;
        height: 6px;
        background: rgba(255,255,255,0.06);
        border-radius: 3px;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #e94560, #ff6b6b);
        transition: width 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }

    .result-pct {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        color: #e94560;
        font-size: 1rem;
        min-width: 58px;
        text-align: right;
    }

    /* File uploader tuning */
    .stFileUploader > div {
        border-color: rgba(233, 69, 96, 0.3) !important;
        border-radius: 12px !important;
    }

    /* Sección de resultados */
    .results-header {
        font-family: 'DM Sans', sans-serif;
        color: #e94560;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 24px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(233,69,96,0.3), transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Cargar modelo (con cache) ───────────────────────────────────
@st.cache_resource
def load_model():
    return torch.jit.load("checkpoints/transfer_exported.pt")

learn_inf = load_model()

# ─── Header ──────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🌍 Landmark Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Sube una imagen de un landmark y descubre cuál es</p>', unsafe_allow_html=True)

# ─── Upload ──────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Arrastra o selecciona una imagen",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    # Mostrar imagen centrada
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Clasificar
    with st.spinner("Analizando imagen..."):
        timg = T.ToTensor()(img).unsqueeze_(0)
        softmax = learn_inf(timg).data.cpu().numpy().squeeze()
        idxs = np.argsort(softmax)[::-1]

    # Resultados
    st.markdown('<div class="results-header">🏆 Top 5 Resultados</div>', unsafe_allow_html=True)

    for i in range(5):
        p = softmax[idxs[i]]
        name = learn_inf.class_names[idxs[i]]
        pct = f"{p * 100:.1f}%"
        width = f"{p * 100:.1f}%"
        rank = i + 1

        st.markdown(f"""
        <div class="result-item">
            <div class="result-rank">{rank}</div>
            <div class="result-info">
                <div class="result-name">{name}</div>
                <div class="bar-track"><div class="bar-fill" style="width:{width}"></div></div>
            </div>
            <div class="result-pct">{pct}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center; color:#555; font-size:12px; font-family:DM Sans">Landmark Explorer · CNN Transfer Learning</p>',
        unsafe_allow_html=True,
    )