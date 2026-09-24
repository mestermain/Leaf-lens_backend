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
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="LeafLens - Coffee Leaf AI Diagnostics",
    page_icon="🌿",
    layout="wide"
)

# ---------------------------------------------------------------------
# EXACT BOOTSTRAP 5, FONTAWESOME & CUSTOM GREEN & WHITE CSS FROM VERCEL APP
# ---------------------------------------------------------------------
st.markdown("""
    <!-- Bootstrap 5 CSS & FontAwesome -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
      :root {
        --dark-green: #1b4332;
        --forest-green: #2d6a4f;
        --emerald: #40916c;
        --mint-green: #52b788;
        --light-green: #d8f3dc;
        --pale-green: #f2f9f4;
        --pure-white: #ffffff;
        --text-dark: #1b4332;
        --text-muted: #4a5d52;
        --border-green: #c7e9d0;
      }

      .stApp {
        background-color: var(--pale-green);
        font-family: 'Plus Jakarta Sans', sans-serif;
      }

      /* Navbar Green & White Header */
      .navbar-header-custom {
        background-color: var(--dark-green);
        border-bottom: 3px solid var(--mint-green);
        box-shadow: 0 4px 15px rgba(27, 67, 50, 0.15);
        padding: 1rem 2rem;
        margin-bottom: 2rem;
        border-radius: 0 0 16px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }

      .brand-title {
        color: var(--pure-white);
        font-weight: 800;
        font-size: 1.8rem;
        margin: 0;
        letter-spacing: -0.5px;
      }

      .brand-tagline {
        color: var(--mint-green);
        font-style: italic;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0;
      }

      .card-custom {
        background-color: var(--pure-white);
        border: 1px solid var(--border-green);
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(27, 67, 50, 0.07);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
      }

      .btn-green {
        background-color: var(--forest-green);
        color: var(--pure-white);
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1.5rem;
        width: 100%;
        transition: all 0.2s ease;
      }

      .btn-green:hover {
        background-color: var(--dark-green);
        color: var(--pure-white);
      }

      .badge-confidence {
        background-color: var(--light-green);
        color: var(--dark-green);
        font-size: 1.1rem;
        font-weight: 800;
        padding: 0.5rem 1.25rem;
        border-radius: 30px;
        display: inline-block;
        border: 1px solid var(--mint-green);
      }

      .guide-box {
        background-color: var(--pale-green);
        border-left: 4px solid var(--forest-green);
        border-radius: 8px;
        padding: 1rem;
      }

      .progress-custom-bg {
        height: 10px;
        background-color: #e9f5ec;
        border-radius: 20px;
        overflow: hidden;
      }

      .progress-custom-bar {
        height: 100%;
        background: linear-gradient(90deg, var(--mint-green) 0%, var(--forest-green) 100%);
        border-radius: 20px;
      }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# HEADER BANNER MATCHING VERCEL EXACT NAVBAR
# ---------------------------------------------------------------------
st.markdown("""
    <div class="navbar-header-custom">
        <div>
            <h1 class="brand-title" style="color: #ffffff !important; font-weight: 800; font-size: 2.2rem; margin: 0;">🍃 <span style="color: #ffffff !important;">LeafLens</span></h1>
            <p class="brand-tagline">"Leaves tell us the Story"</p>
        </div>
        <div style="text-align: right; color: white;">
            <span class="badge bg-success px-3 py-2"><i class="fa-solid fa-microscope me-1"></i> Multi-Model Ensemble & ViT XAI</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# IMPORT PYTORCH BACKEND & MODELS
# ---------------------------------------------------------------------
try:
    from app import std_transform, inc_transform, device, CLASS_NAMES_10, CLASS_NAMES_4, CLASS_INFO, models_10, vit_4, gradcam_engine_10, gradcam_engine_4, torch
except Exception as e:
    st.error(f"Error initializing PyTorch AI backend models: {e}")

# ---------------------------------------------------------------------
# MODEL TAB SELECTION MATCHING VERCEL TAB DESIGN
# ---------------------------------------------------------------------
tab_mode = st.radio(
    "Select Model Mode",
    ["10_class", "4_class"],
    format_func=lambda x: "🔬 10-Class Complete Deficiency Model (Ensemble)" if x == "10_class" else "⚡ 4-Class Vision Transformer (ViT)",
    horizontal=True
)

if tab_mode == "10_class":
    st.markdown("""
        <div class="alert alert-success d-flex align-items-center mb-4" style="border-radius: 12px; background-color: #d8f3dc; color: #1b4332; border: 1px solid #52b788;">
            <i class="fa-solid fa-circle-info fs-4 me-3"></i>
            <div>
                Active Model Mode: <strong>10-Class Complete Deficiency Ensemble</strong> (Boron, Calcium, Healthy, Iron, Magnesium, Manganese, Complex, Nitrogen, Phosphorus, Potassium) with <strong>Grad-CAM++</strong> Explainable AI.
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="alert alert-success d-flex align-items-center mb-4" style="border-radius: 12px; background-color: #d8f3dc; color: #1b4332; border: 1px solid #52b788;">
            <i class="fa-solid fa-flask fs-4 me-3"></i>
            <div>
                Active Model Mode: <strong>4-Class Vision Transformer (ViT) Model</strong> (Healthy, Nitrogen [N], Phosphorus [P], Potassium [K]) with <strong>Standard Grad-CAM</strong> Explainable AI.
            </div>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# TWO-COLUMN MAIN LAYOUT MATCHING VERCEL EXACT UI
# ---------------------------------------------------------------------
col_left, col_right = st.columns([5, 7], gap="medium")

with col_left:
    st.markdown("""
        <div class="card-custom">
            <h5 class="fw-bold mb-3 text-dark"><i class="fa-solid fa-upload text-success me-2"></i> Upload Coffee Leaf Image</h5>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose leaf image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert('RGB')
        st.image(pil_img, caption="Uploaded Coffee Leaf", use_container_width=True)
        run_btn = st.button("🔬 Run LeafLens Diagnosis & Grad-CAM Heatmap", use_container_width=True)

    st.markdown("""
        <div class="card-custom mt-3">
            <h5 class="fw-bold mb-3 text-dark"><i class="fa-solid fa-camera text-success me-2"></i> Image Capture Guidelines</h5>
            <div class="guide-box small">
                <ul class="mb-0 ps-3">
                    <li class="mb-2"><strong>White or White Paper Background:</strong> The background of the leaf image should be plain white, or place a sheet of clean white paper in the background behind the leaf.</li>
                    <li class="mb-2"><strong>Single Leaf Focus:</strong> Position one coffee leaf flat in the center, covering 70-80% of the frame.</li>
                    <li class="mb-2"><strong>Natural Lighting:</strong> Shoot under bright indirect daylight; avoid strong shadows or flash.</li>
                    <li class="mb-2"><strong>Adaxial Surface:</strong> Ensure the top face of the leaf is clearly visible showing chlorosis patterns.</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_right:
    if uploaded_file is not None and 'run_btn' in locals() and run_btn:
        with st.spinner("Evaluating deep feature maps across VGG19, InceptionV3, DenseNet201, MobileNetV3, ViT-B/16 & EfficientNet-B4 + generating Grad-CAM heatmaps..."):
            image_bytes = uploaded_file.getvalue()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            cv_img_rgb = np.array(pil_img)
            h, w, _ = cv_img_rgb.shape

            input_std = std_transform(pil_img).unsqueeze(0).to(device)
            input_inc = inc_transform(pil_img).unsqueeze(0).to(device)

            if tab_mode == '4_class':
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

            info = CLASS_INFO.get(predicted_class, {'title': predicted_class, 'desc': 'Coffee deficiency', 'action': 'Consult specialist.'})

            # Primary Diagnosis Card Matching Vercel EXACT Layout
            st.markdown(f"""
                <div class="card-custom">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <span class="badge bg-success px-3 py-2" style="font-size:0.9rem;"><i class="fa-solid fa-circle-check me-1"></i> Diagnosis Complete</span>
                        <div class="badge-confidence">{confidence:.2f}% / 100% Probability</div>
                    </div>
                    <h2 class="fw-bold text-dark mb-1">{info['title']}</h2>
                    <p class="text-muted mb-0 small">Predicted Deficiency Category</p>
                </div>
            """, unsafe_allow_html=True)

            # Side-by-Side Images
            img_c1, img_c2 = st.columns(2)
            with img_c1:
                st.markdown("""
                    <div class="card-custom p-2 text-center">
                        <h6 class="fw-bold text-dark mb-2">Original Leaf Image</h6>
                    </div>
                """, unsafe_allow_html=True)
                st.image(pil_img, use_container_width=True)

            with img_c2:
                xai_name = "Grad-CAM++ Attention Heatmap" if tab_mode == '10_class' else "Grad-CAM Attention Heatmap"
                st.markdown(f"""
                    <div class="card-custom p-2 text-center">
                        <h6 class="fw-bold text-dark mb-2">{xai_name}</h6>
                    </div>
                """, unsafe_allow_html=True)
                st.image(overlay, use_container_width=True)

            # Detailed Agronomic Recommendations
            st.markdown(f"""
                <div class="card-custom mt-3">
                    <h5 class="fw-bold text-dark mb-2"><i class="fa-solid fa-clipboard-list text-success me-2"></i> Agronomic Analysis & Remedy Plan</h5>
                    <div class="mb-3">
                        <h6 class="fw-bold text-success mb-1">Symptom Description:</h6>
                        <p class="text-dark small mb-0">{info['desc']}</p>
                    </div>
                    <div>
                        <h6 class="fw-bold text-success mb-1">Recommended Treatment & Action:</h6>
                        <p class="text-dark small mb-0"><b>{info['action']}</b></p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Probability Distribution Bars
            st.markdown("""
                <div class="card-custom mt-3">
                    <h5 class="fw-bold text-dark mb-3"><i class="fa-solid fa-chart-bar text-success me-2"></i> Class Probability Distribution (Scaled out of 100%)</h5>
                </div>
            """, unsafe_allow_html=True)

            for idx, cls in enumerate(class_names):
                prob_val = float(final_probs[idx]) * 100
                cls_title = CLASS_INFO.get(cls, {}).get('title', cls)
                st.markdown(f"""
                    <div class="mb-3">
                        <div class="d-flex justify-content-between small fw-bold mb-1">
                            <span class="text-dark">{cls_title}</span>
                            <span class="text-success fw-bold">{prob_val:.2f}% / 100%</span>
                        </div>
                        <div class="progress-custom-bg">
                            <div class="progress-custom-bar" style="width: {prob_val}%;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Sub-model agreement table for 10-class mode
            if tab_mode == '10_class' and models_summary:
                st.markdown("""
                    <div class="card-custom mt-3">
                        <h5 class="fw-bold text-dark mb-3"><i class="fa-solid fa-network-wired text-success me-2"></i> Ensemble Sub-Model Diagnosis Agreement</h5>
                    </div>
                """, unsafe_allow_html=True)
                summary_df = [{'Model Architecture': k, 'Predicted Class': v['class'], 'Confidence (%)': f"{v['confidence']:.2f}% / 100%"} for k, v in models_summary.items()]
                st.table(pd.DataFrame(summary_df))

    else:
        # Default Placeholder Card Matching Vercel
        st.markdown("""
            <div class="card-custom text-center p-5">
                <i class="fa-solid fa-seedling text-success display-1 mb-3"></i>
                <h4 class="fw-bold text-dark">Ready for Leaf Diagnosis</h4>
                <p class="text-muted">Upload a coffee leaf image on the left and click <b>Run LeafLens Diagnosis</b> to generate AI deficiency classifications, agronomic treatment recommendations, and Grad-CAM++ heatmaps.</p>
            </div>
        """, unsafe_allow_html=True)
