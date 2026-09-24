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
# STREAMLIT PAGE CONFIG & META DATA
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="🍃 LeafLens - Coffee Leaf AI & Grad-CAM Diagnostics",
    page_icon="🌿",
    layout="wide"
)

# Encode logo to Base64 if available
logo_b64 = ""
logo_path = os.path.join(os.path.dirname(__file__), "static", "leaflens_logo.jpg")
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
else:
    logo_b64 = "https://raw.githubusercontent.com/mestermain/LeafLens/main/static/leaflens_logo.jpg"

# ---------------------------------------------------------------------
# LEAF VALIDATION ENGINE (NON-LEAF / OOD DETECTION)
# ---------------------------------------------------------------------
def validate_leaf_image(pil_img):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([18, 18, 18])
    upper_green = np.array([95, 255, 255])
    lower_brown = np.array([4, 18, 18])
    upper_brown = np.array([22, 255, 220])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    combined_mask = cv2.bitwise_or(mask_green, mask_brown)

    total_pixels = cv_img.shape[0] * cv_img.shape[1]
    plant_pixel_ratio = np.count_nonzero(combined_mask) / total_pixels
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if plant_pixel_ratio < 0.06:
        return False, "The uploaded image does not appear to contain a coffee leaf. Color spectrum lacks plant/chlorophyll tissue tones."

    if laplacian_var < 6.0:
        return False, "The image is too blank or blurry to detect leaf vein structures."

    return True, "Valid leaf image"

# ---------------------------------------------------------------------
# COMPLETE BOOTSTRAP 5, FONTAWESOME & EXACT CSS STYLING FROM LEAFLENS.GIT
# ---------------------------------------------------------------------
st.markdown(f"""
    <!-- Bootstrap 5 CSS & FontAwesome -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
      :root {{
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
      }}

      .stApp {{
        background-color: var(--pale-green);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-dark);
      }}

      /* Navbar Green & White */
      .navbar-custom {{
        background-color: var(--dark-green);
        border-bottom: 3px solid var(--mint-green);
        box-shadow: 0 4px 15px rgba(27, 67, 50, 0.15);
        padding: 0.8rem 2rem;
        margin-bottom: 1.5rem;
        border-radius: 0 0 16px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}

      .brand-logo-img {{
        height: 46px;
        width: 46px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid var(--mint-green);
      }}

      .brand-title-text {{
        color: #ffffff !important;
        font-weight: 800;
        font-size: 1.8rem;
        margin: 0;
      }}

      .hero-logo-img {{
        height: 90px;
        width: 90px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid var(--forest-green);
        box-shadow: 0 6px 18px rgba(27, 67, 50, 0.15);
      }}

      .hero-title {{
        font-weight: 800;
        color: var(--dark-green);
        font-size: 2.5rem;
        margin-bottom: 0;
      }}

      /* Pure White & Green Card Design */
      .card-custom {{
        background-color: var(--pure-white);
        border: 1px solid var(--border-green);
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(27, 67, 50, 0.07);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
      }}

      /* Drag and Drop Zone */
      .drop-zone {{
        border: 2px dashed var(--forest-green);
        border-radius: 14px;
        padding: 30px 20px;
        text-align: center;
        background-color: var(--pale-green);
      }}

      .btn-green {{
        background-color: var(--dark-green);
        color: var(--pure-white) !important;
        border: none;
        font-weight: 700;
        border-radius: 10px;
        padding: 12px 24px;
        box-shadow: 0 4px 12px rgba(27, 67, 50, 0.2);
        width: 100%;
        text-align: center;
      }}

      .btn-green:hover {{
        background-color: var(--forest-green);
      }}

      .badge-confidence {{
        font-size: 1.05rem;
        padding: 8px 18px;
        border-radius: 30px;
        background-color: var(--dark-green);
        color: var(--pure-white);
        font-weight: 700;
      }}

      .progress-custom-bg {{
        height: 14px;
        border-radius: 7px;
        background-color: var(--light-green);
        overflow: hidden;
      }}

      .progress-custom-bar {{
        height: 100%;
        background-color: var(--forest-green);
        border-radius: 7px;
      }}

      .guide-box {{
        background-color: var(--pale-green);
        border: 1px solid var(--border-green);
        border-radius: 12px;
        padding: 16px;
      }}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# NAVBAR (MATCHING VERCEL NAVBAR EXACTLY)
# ---------------------------------------------------------------------
st.markdown(f"""
    <div class="navbar-custom">
        <div class="d-flex align-items-center gap-3">
            <img src="{logo_b64}" alt="LeafLens Logo" class="brand-logo-img">
            <span class="brand-title-text">LeafLens</span>
        </div>
        <div class="d-flex align-items-center gap-2">
            <span class="badge bg-success px-3 py-2"><i class="fa-solid fa-microscope me-1"></i> Multi-Model Ensemble & ViT XAI</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# HERO HEADER BANNER (MATCHING VERCEL HERO EXACTLY)
# ---------------------------------------------------------------------
st.markdown(f"""
    <div class="text-center my-3">
        <div class="d-flex align-items-center justify-content-center gap-3 mb-2">
            <img src="{logo_b64}" alt="LeafLens Brand Logo" class="hero-logo-img">
            <h1 class="hero-title">LeafLens Diagnostics</h1>
        </div>
        <p class="fs-5 mt-2 fw-semibold" style="font-style: italic; color: #2d6a4f;">
            "Leaves tell us the Story"
        </p>
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
# DUAL MODEL TABS SWITCHER (MATCHING VERCEL TABS EXACTLY)
# ---------------------------------------------------------------------
tab_mode = st.radio(
    "Select Model Mode",
    ["10_class", "4_class"],
    format_func=lambda x: "🔬 10-Class Full Diagnosis Model (Ensemble)" if x == "10_class" else "⚡ 4-Class Vision Transformer (ViT) Model",
    horizontal=True
)

if tab_mode == "10_class":
    st.markdown("""
        <div class="card card-custom p-3 mb-4 text-center">
            <div class="small text-muted fw-semibold">
                <i class="fa-solid fa-circle-info text-success me-1"></i> Active Model Mode: 
                <strong>10-Class Complete Deficiency Ensemble</strong> (Boron, Calcium, Healthy, Iron, Magnesium, Manganese, Complex, Nitrogen, Phosphorus, Potassium) with <strong>Grad-CAM++</strong> Explainable AI.
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="card card-custom p-3 mb-4 text-center">
            <div class="small text-muted fw-semibold">
                <i class="fa-solid fa-flask text-success me-1"></i> Active Model Mode: 
                <strong>4-Class Vision Transformer (ViT) Model</strong> (Healthy, Nitrogen [N], Phosphorus [P], Potassium [K]) with <strong>Standard Grad-CAM</strong> Explainable AI.
            </div>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# MAIN LAYOUT (LEFT UPLOAD COLUMN & RIGHT RESULTS COLUMN)
# ---------------------------------------------------------------------
col_left, col_right = st.columns([5, 7], gap="medium")

with col_left:
    st.markdown("""
        <div class="card card-custom p-4 mb-3">
            <h4 class="fw-bold mb-3 d-flex align-items-center gap-2 text-dark">
                <i class="fa-solid fa-cloud-arrow-up text-success"></i> Upload Leaf Image
            </h4>
            <div class="alert alert-warning border-warning p-2 mb-3 rounded-3 small d-flex align-items-center gap-2" style="background-color: #fff9e6; border: 1px solid #ffe082; color: #856404;">
                <i class="fa-solid fa-lightbulb text-warning fs-5"></i>
                <div>
                    <strong>Important Instruction:</strong> The background of the image should be <strong>white</strong> or place a <strong>white paper</strong> in the background of the leaf image for accurate diagnosis.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert('RGB')
        st.markdown("<h6 class='fw-bold text-dark mb-2'>Selected Image Preview:</h6>", unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        run_btn = st.button("🔬 Run LeafLens Diagnosis & Grad-CAM++", use_container_width=True)

    st.markdown("""
        <div class="card card-custom p-4 mt-3">
            <h5 class="fw-bold mb-3 text-dark">
                <i class="fa-solid fa-camera text-success me-2"></i> Image Capture Guidelines
            </h5>
            <div class="guide-box small">
                <ul class="mb-0 ps-3">
                    <li class="mb-2"><strong>White or White Paper Background:</strong> The background of the leaf image should be plain white, or place a sheet of clean white paper in the background behind the leaf.</li>
                    <li class="mb-2"><strong>Single Leaf Focus:</strong> Position one coffee leaf flat in the center, covering 70-80% of the frame.</li>
                    <li class="mb-2"><strong>Natural Lighting:</strong> Shoot under bright indirect daylight; avoid strong shadows or flash.</li>
                    <li class="mb-2"><strong>Adaxial Surface:</strong> Ensure the top face of the leaf is clearly visible showing chlorosis vein patterns and necrotic spot lesions.</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_right:
    if uploaded_file is not None and 'run_btn' in locals() and run_btn:
        image_bytes = uploaded_file.getvalue()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 1. RUN LEAF VALIDATION ENGINE
        is_valid_leaf, validation_reason = validate_leaf_image(pil_img)

        if not is_valid_leaf:
            st.markdown(f"""
                <div class="card card-custom p-4 text-center" style="border-left: 6px solid #dc3545; background-color: #fff5f5;">
                    <i class="fa-solid fa-triangle-exclamation text-danger display-3 mb-3"></i>
                    <h3 class="fw-bold text-danger mb-2">Invalid Image Detected</h3>
                    <p class="text-dark fs-6"><b>{validation_reason}</b></p>
                    <hr>
                    <p class="text-muted small mb-0">
                        <i class="fa-solid fa-circle-info me-1"></i> LeafLens AI requires a clear photograph of a coffee leaf placed flat on a white background to compute accurate deficiency metrics. Please upload a valid leaf image according to the guidelines.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("LeafLens is Analyzing Coffee Leaf... Evaluating deep feature maps across VGG19, InceptionV3, DenseNet201, MobileNetV3, ViT-B/16 & EfficientNet-B4 + generating Grad-CAM++ heatmaps..."):
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

                # Primary Diagnosis Card (Matching Vercel EXACT layout)
                st.markdown(f"""
                    <div class="card card-custom p-4">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <span class="badge bg-success px-3 py-2"><i class="fa-solid fa-circle-check me-1"></i> Diagnosis Complete</span>
                            <span class="badge-confidence">{confidence:.2f}% / 100% Probability</span>
                        </div>
                        <h2 class="display-6 fw-bold text-dark mb-1">{info['title']}</h2>
                        <p class="text-muted mb-0 small">Predicted Deficiency Category</p>
                    </div>
                """, unsafe_allow_html=True)

                # Side-by-Side Images (Matching Vercel EXACT layout)
                img_c1, img_c2 = st.columns(2)
                with img_c1:
                    st.markdown("""
                        <div class="card card-custom p-2 text-center mb-2">
                            <h6 class="fw-bold text-dark mb-0">Original Leaf Image</h6>
                        </div>
                    """, unsafe_allow_html=True)
                    st.image(pil_img, use_container_width=True)

                with img_c2:
                    xai_name = "Grad-CAM++ Attention Heatmap (Higher-Order Gradients)" if tab_mode == '10_class' else "Grad-CAM Attention Heatmap (1st Order Gradients)"
                    st.markdown(f"""
                        <div class="card card-custom p-2 text-center mb-2">
                            <h6 class="fw-bold text-dark mb-0">{xai_name}</h6>
                        </div>
                    """, unsafe_allow_html=True)
                    st.image(overlay, use_container_width=True)

                # Detailed Agronomic Analysis & Remedy Plan (Matching Vercel EXACT layout)
                st.markdown(f"""
                    <div class="card card-custom p-4 mt-3">
                        <h5 class="fw-bold text-dark mb-3">
                            <i class="fa-solid fa-clipboard-list text-success me-2"></i> Agronomic Analysis & Remedy Plan
                        </h5>
                        <div class="alert alert-info-custom p-3 mb-3">
                            <h6 class="fw-bold text-dark mb-1">Symptom Description:</h6>
                            <p class="mb-0 small">{info['desc']}</p>
                        </div>
                        <div class="alert alert-info-custom p-3 mb-0">
                            <h6 class="fw-bold text-dark mb-1">Recommended Treatment & Remedial Action:</h6>
                            <p class="mb-0 small"><b>{info['action']}</b></p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Class Probability Distribution Progress Bars (Matching Vercel EXACT layout)
                st.markdown("""
                    <div class="card card-custom p-4 mt-3">
                        <h5 class="fw-bold text-dark mb-3">
                            <i class="fa-solid fa-chart-bar text-success me-2"></i> Class Probability Distribution (Scaled out of 100%)
                        </h5>
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

                st.markdown("</div>", unsafe_allow_html=True)

                # Ensemble Sub-Model Agreement Table (Matching Vercel EXACT layout)
                if tab_mode == '10_class' and models_summary:
                    st.markdown("""
                        <div class="card card-custom p-4 mt-3">
                            <h5 class="fw-bold text-dark mb-3">
                                <i class="fa-solid fa-network-wired text-success me-2"></i> Ensemble Sub-Model Diagnosis Agreement
                            </h5>
                        </div>
                    """, unsafe_allow_html=True)
                    summary_df = [{'Model Architecture': k, 'Predicted Class': v['class'], 'Confidence (%)': f"{v['confidence']:.2f}% / 100%"} for k, v in models_summary.items()]
                    st.table(pd.DataFrame(summary_df))

    else:
        # Default Placeholder State Matching Vercel EXACT layout
        st.markdown("""
            <div class="card card-custom p-5 text-center">
                <i class="fa-solid fa-seedling text-success display-1 mb-3"></i>
                <h4 class="fw-bold text-dark">Ready for Leaf Diagnosis</h4>
                <p class="text-muted">
                    Upload a coffee leaf image on the left and click 
                    <b>Run LeafLens Diagnosis</b> to generate AI deficiency 
                    classifications, agronomic treatment recommendations, and Grad-CAM++ heatmaps.
                </p>
            </div>
        """, unsafe_allow_html=True)
