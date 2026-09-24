# 🌿 LeafLens AI Model Server (Backend API)

This repository contains the dedicated **PyTorch Model Backend API** for the **LeafLens** Coffee Leaf Deficiency AI Application.

## 🚀 Features
- **10-Class Multi-Model Ensemble**: VGG19-BN, InceptionV3, DenseNet201, MobileNetV3-Large, ViT-B/16, EfficientNet-B4, and Stacking Meta-Learner.
- **4-Class Vision Transformer (ViT)**: Dedicated HuggingFace/PyTorch ViT model.
- **Explainable AI (XAI)**:
  - **Grad-CAM++** for 10-Class Ensemble.
  - **Standard Grad-CAM** for 4-Class ViT patch tokens.
- **CORS Enabled**: Ready to serve frontend requests from Vercel or any web application.

## 🔌 API Endpoints
- `GET /health`: Returns backend service health status and loaded model list.
- `POST /predict`: Accepts leaf image upload + `model_mode` (`10_class` or `4_class`), returning predicted deficiency, probability distribution, agronomic guidance, and base64-encoded Grad-CAM heatmap visualization.

## 🛠️ Deployment
Can be deployed to Render, Railway, Hugging Face Spaces, Streamlit Community Cloud, or any Linux server instance.
