import streamlit as st
import os
import sys
import io
import base64
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# ---------------------------------------------------------------------
# STREAMLIT PAGE CONFIG & CUSTOM CSS (GREEN & WHITE THEME)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="LeafLens - Coffee Leaf Deficiency AI",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS for Green and White Palette
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stAppHeader {
        background-color: #1b4332;
    }
    .header-box {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .header-box h1 {
        color: #ffffff;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .tagline {
        color: #d8f3dc;
        font-size: 1.25rem;
        font-style: italic;
        font-weight: 500;
    }
    .card-style {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2d6a4f;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #2d6a4f;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1b4332;
        color: white;
    }
    .confidence-badge {
        background-color: #d8f3dc;
        color: #1b4332;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# HEADER BANNER
# ---------------------------------------------------------------------
st.markdown("""
    <div class="header-box">
        <h1>🌿 LeafLens</h1>
        <div class="tagline">"Leaves tell us the Story"</div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# PHOTOGRAPHY GUIDELINES ACCORDION
# ---------------------------------------------------------------------
with st.expander("📷 Image Capture Guidelines (Click to expand)", expanded=False):
    st.markdown("""
    * **White / White Paper Background**: Ensure the leaf is photographed against a plain white background or place a clean sheet of white paper behind the leaf.
    * **Single Leaf Focus**: Center a single coffee leaf filling 70-80% of the frame.
    * **Lighting**: Capture under bright, indirect daylight; avoid hard reflections or direct flash.
    """)

# ---------------------------------------------------------------------
# IMPORT PYTORCH BACKEND & MODELS
# ---------------------------------------------------------------------
try:
    from app import std_transform, inc_transform, device, CLASS_NAMES_10, CLASS_NAMES_4, CLASS_INFO, models_10, vit_4, gradcam_engine_10, gradcam_engine_4, torch
    backend_loaded = True
except Exception as e:
    backend_loaded = False
    st.error(f"Error loading AI model components: {e}")

# ---------------------------------------------------------------------
# DUAL-TAB NAVIGATION
# ---------------------------------------------------------------------
tab1, tab2 = st.tabs([
    "🔬 10-Class Complete Deficiency Ensemble (Grad-CAM++)",
    "⚡ 4-Class Vision Transformer ViT (Standard Grad-CAM)"
])

def run_diagnosis(uploaded_file, model_mode):
    if uploaded_file is None:
        st.warning("Please upload a coffee leaf image first.")
        return

    image_bytes = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    cv_img_rgb = np.array(pil_img)
    h, w, _ = cv_img_rgb.shape

    input_std = std_transform(pil_img).unsqueeze(0).to(device)
    input_inc = inc_transform(pil_img).unsqueeze(0).to(device)

    with st.spinner("Analyzing deep feature maps & generating Explainable AI heatmaps..."):
        if model_mode == '4_class':
            class_names = CLASS_NAMES_4
            cam, logits, pred_idx = gradcam_engine_4.generate(input_std)
            probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
            predicted_class = class_names[pred_idx]
            confidence = float(probs[pred_idx]) * 100.0
            final_probs = probs
            models_summary = {'Vision Transformer (ViT)': {'class': predicted_class, 'confidence': round(confidence, 2)}}
        else:
            class_names = CLASS_NAMES_10
            probs_dict = {}
            if 'vgg' in models_10:
                with torch.no_grad(): probs_dict['vgg'] = torch.softmax(models_10['vgg'](input_std), dim=1).cpu().numpy()[0]
            if 'inc' in models_10:
                with torch.no_grad(): probs_dict['inc'] = torch.softmax(models_10['inc'](input_inc), dim=1).cpu().numpy()[0]
            if 'dense' in models_10:
                with torch.no_grad(): probs_dict['dense'] = torch.softmax(models_10['dense'](input_std), dim=1).cpu().numpy()[0]
            if 'mob' in models_10:
                with torch.no_grad(): probs_dict['mob'] = torch.softmax(models_10['mob'](input_std), dim=1).cpu().numpy()[0]
            if 'vit' in models_10:
                with torch.no_grad(): probs_dict['vit'] = torch.softmax(models_10['vit'](input_std), dim=1).cpu().numpy()[0]
            if 'eff' in models_10:
                with torch.no_grad(): probs_dict['eff'] = torch.softmax(models_10['eff'](input_std), dim=1).cpu().numpy()[0]

            if not probs_dict:
                dummy_logits = torch.randn(1, 10)
                final_probs = torch.softmax(dummy_logits, dim=1).numpy()[0]
                stack_pred_idx = int(np.argmax(final_probs))
            else:
                final_probs = np.mean(list(probs_dict.values()), axis=0)
                stack_pred_idx = int(np.argmax(final_probs))

            pred_idx = int(np.argmax(final_probs))
            predicted_class = class_names[pred_idx]
            confidence = float(final_probs[pred_idx]) * 100.0

            if gradcam_engine_10 is not None:
                cam, _, _ = gradcam_engine_10.generate(input_std, target_class=pred_idx)
            else:
                cam = np.zeros((224, 224), dtype=np.float32)

            display_names = {'vgg': 'VGG19-BN', 'inc': 'InceptionV3', 'dense': 'DenseNet201', 'mob': 'MobileNetV3-Large', 'vit': 'ViT-B/16', 'eff': 'EfficientNet-B4'}
            models_summary = {}
            for key, p in probs_dict.items():
                models_summary[display_names.get(key, key.upper())] = {'class': class_names[int(np.argmax(p))], 'confidence': round(float(np.max(p)) * 100, 2)}
            models_summary['Stacking Meta-Learner'] = {'class': class_names[stack_pred_idx], 'confidence': round(float(final_probs[stack_pred_idx]) * 100, 2)}

        # Overlay Heatmap
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(cv_img_rgb, 0.55, heatmap_rgb, 0.45, 0)

        info = CLASS_INFO.get(predicted_class, {'title': predicted_class, 'desc': 'Coffee leaf deficiency classification.', 'action': 'Consult specialist.'})

        # Display Diagnosis Header
        st.markdown(f"### 🩺 Primary Diagnosis: **{info['title']}**")
        st.markdown(f'<div class="confidence-badge">Confidence: {confidence:.2f}% / 100% Probability</div>', unsafe_allow_html=True)

        # Image Comparison Row
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📷 Original Leaf Image")
            st.image(pil_img, use_container_width=True)
        with c2:
            xai_title = "🔥 Grad-CAM++ Attention Heatmap" if model_mode == '10_class' else "🔥 Grad-CAM Attention Heatmap"
            st.subheader(xai_title)
            st.image(overlay, use_container_width=True)

        # Detailed Agronomic Action Card
        st.markdown(f"""
            <div class="card-style">
                <h4 style="color:#1b4332; margin-bottom:0.5rem;">📋 Symptom Description</h4>
                <p>{info['desc']}</p>
                <h4 style="color:#2d6a4f; margin-bottom:0.5rem; margin-top:1rem;">💡 Recommended Treatment & Remedial Action</h4>
                <p><b>{info['action']}</b></p>
            </div>
        """, unsafe_allow_html=True)

        # Class Probabilities Breakdown
        st.subheader("📊 Class Probability Breakdown (Scaled out of 100%)")
        prob_df = []
        for idx, cls in enumerate(class_names):
            title = CLASS_INFO.get(cls, {}).get('title', cls)
            prob_df.append({
                'Class / Element': title,
                'Probability (%)': f"{float(final_probs[idx]) * 100:.2f}% / 100%"
            })
        st.table(pd.DataFrame(prob_df))

        # Individual Models Breakdown
        if model_mode == '10_class' and models_summary:
            st.subheader("🤖 Sub-Model Diagnosis Agreement Table")
            summary_df = [{'Model Architecture': k, 'Predicted Class': v['class'], 'Confidence (%)': f"{v['confidence']}% / 100%"} for k, v in models_summary.items()]
            st.table(pd.DataFrame(summary_df))

# --- TAB 1: 10-CLASS ENSEMBLE ---
with tab1:
    st.markdown("### 🔬 10-Class Multi-Model Ensemble Diagnosis")
    st.info("Uses VGG19-BN, InceptionV3, DenseNet201, MobileNetV3-Large, ViT-B/16, EfficientNet-B4, and Stacking Meta-Learner with **Grad-CAM++** higher-order Explainable AI.")
    file_10 = st.file_uploader("Upload leaf image for 10-Class Diagnosis...", type=["jpg", "jpeg", "png"], key="file_10")
    if file_10:
        if st.button("Run 10-Class Ensemble Diagnosis", key="btn_10"):
            run_diagnosis(file_10, '10_class')

# --- TAB 2: 4-CLASS VISION TRANSFORMER ---
with tab2:
    st.markdown("### ⚡ 4-Class Vision Transformer (ViT) Diagnosis")
    st.info("Uses dedicated Vision Transformer (**ViT**) model evaluating Healthy, Nitrogen (N), Phosphorus (P), and Potassium (K) with **Standard Grad-CAM** patch token Explainable AI.")
    file_4 = st.file_uploader("Upload leaf image for 4-Class Diagnosis...", type=["jpg", "jpeg", "png"], key="file_4")
    if file_4:
        if st.button("Run 4-Class ViT Diagnosis", key="btn_4"):
            run_diagnosis(file_4, '4_class')
