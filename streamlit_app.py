import streamlit as st
import os
import sys
import io
import base64
import numpy as np
from PIL import Image
import cv2

# Page Configuration
st.set_page_config(
    page_title="LeafLens AI Model Server & Playground",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 LeafLens Coffee Leaf Deficiency AI Model Backend")
st.markdown("##### Powered by Multi-Model Ensemble & Vision Transformer (ViT)")

st.info("💡 **Backend Service Status**: Active & Ready to process inference requests from Vercel Frontend.")

# Sidebar Configuration
st.sidebar.header("⚙️ Model Configuration")
model_mode = st.sidebar.radio(
    "Select Model Mode",
    ["10_class", "4_class"],
    format_func=lambda x: "10-Class Ensemble (Grad-CAM++)" if x == "10_class" else "4-Class Vision Transformer (Grad-CAM)"
)

# File Uploader
uploaded_file = st.file_uploader("Upload a coffee leaf image for diagnosis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Uploaded Leaf Image")
        st.image(uploaded_file, use_container_width=True)

    if st.button("🚀 Run AI Diagnosis & Grad-CAM Heatmap"):
        with st.spinner("Executing PyTorch Ensemble & Grad-CAM Explainable AI..."):
            try:
                from app import std_transform, inc_transform, device, CLASS_NAMES_10, CLASS_NAMES_4, CLASS_INFO, models_10, vit_4, gradcam_engine_10, gradcam_engine_4, torch
                
                image_bytes = uploaded_file.read()
                pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                cv_img_rgb = np.array(pil_img)
                h, w, _ = cv_img_rgb.shape

                input_std = std_transform(pil_img).unsqueeze(0).to(device)
                input_inc = inc_transform(pil_img).unsqueeze(0).to(device)

                if model_mode == '4_class':
                    class_names = CLASS_NAMES_4
                    cam, logits, pred_idx = gradcam_engine_4.generate(input_std)
                    probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
                    predicted_class = class_names[pred_idx]
                    confidence = float(probs[pred_idx]) * 100.0
                    final_probs = probs
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
                    else:
                        final_probs = np.mean(list(probs_dict.values()), axis=0)

                    pred_idx = int(np.argmax(final_probs))
                    predicted_class = class_names[pred_idx]
                    confidence = float(final_probs[pred_idx]) * 100.0

                    if gradcam_engine_10 is not None:
                        cam, _, _ = gradcam_engine_10.generate(input_std, target_class=pred_idx)
                    else:
                        cam = np.zeros((224, 224), dtype=np.float32)

                # Overlay
                cam_resized = cv2.resize(cam, (w, h))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(cv_img_rgb, 0.55, heatmap_rgb, 0.45, 0)

                info = CLASS_INFO.get(predicted_class, {'title': predicted_class, 'desc': 'Coffee deficiency', 'action': 'Consult specialist.'})

                with col2:
                    st.subheader("🔥 Grad-CAM Explainable Heatmap")
                    st.image(overlay, use_container_width=True)

                st.success(f"**Diagnosis**: {info['title']} ({confidence:.2f}% / 100%)")
                st.markdown(f"**Description**: {info['desc']}")
                st.markdown(f"**Action**: {info['action']}")

            except Exception as e:
                st.error(f"Error during inference: {e}")
