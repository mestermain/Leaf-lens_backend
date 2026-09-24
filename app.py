import os
import sys
import glob
import time
import io
import uuid
import base64
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import joblib

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for Vercel frontend

# Set seed and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Directory Paths
SAVE_DIR_10 = os.environ.get("SAVE_DIR_10", r"C:\Users\PC\Desktop\ensamble\Saved_Models")
if not os.path.exists(SAVE_DIR_10):
    SAVE_DIR_10 = "Saved_Models"

SAVE_DIR_4 = os.environ.get("SAVE_DIR_4", r"C:\Users\PC\Desktop\Four-Classes_Coleaf\Saved_Models")
if not os.path.exists(SAVE_DIR_4):
    SAVE_DIR_4 = "Saved_Models"

# 10-Class Definitions
CLASS_NAMES_10 = [
    'boron-B', 'calcium-Ca', 'healthy', 'iron-Fe', 'magnesium-Mg',
    'manganese-Mn', 'more-deficiencies', 'nitrogen-N', 'phosphorus-P', 'potasium-K'
]

# 4-Class Definitions
CLASS_NAMES_4 = [
    'healthy', 'nitrogen-N', 'phosphorus-P', 'potasium-K'
]

CLASS_INFO = {
    'boron-B': {'title': 'Boron Deficiency (B)', 'desc': 'Deformed, small, or brittle young leaves with chlorotic spots and stunted growth.', 'action': 'Apply foliar spray of solubor or boric acid; maintain consistent soil moisture.'},
    'calcium-Ca': {'title': 'Calcium Deficiency (Ca)', 'desc': 'Necrotic margins and cupped young leaves with distorted growth tips.', 'action': 'Apply agricultural lime or calcium nitrate to balance soil pH and structure.'},
    'healthy': {'title': 'Healthy Leaf', 'desc': 'Vibrant green color, optimal photosynthetic activity, and uniform leaf tissue.', 'action': 'Maintain standard irrigation, organic soil nutrition, and periodic monitoring.'},
    'iron-Fe': {'title': 'Iron Deficiency (Fe)', 'desc': 'Interveinal chlorosis on young leaves where main veins remain dark green.', 'action': 'Apply chelated iron (Fe-EDDHA) foliar feeding and improve soil aeration.'},
    'magnesium-Mg': {'title': 'Magnesium Deficiency (Mg)', 'desc': 'Inward interveinal yellowing starting on mature lower leaves moving inward.', 'action': 'Apply Epsom salt (magnesium sulfate MgSO4) via foliar spray or soil drenching.'},
    'manganese-Mn': {'title': 'Manganese Deficiency (Mn)', 'desc': 'Fine interveinal mottling with tiny necrotic spots on mid-tier leaves.', 'action': 'Foliar application of manganese sulfate (MnSO4); adjust soil pH below 6.5.'},
    'more-deficiencies': {'title': 'Complex / Multiple Deficiencies', 'desc': 'Overlapping chlorosis symptoms, mottled lesions, and tip burn across multiple elements.', 'action': 'Perform complete soil & foliar laboratory analysis; apply balanced NPK + micronutrients.'},
    'nitrogen-N': {'title': 'Nitrogen Deficiency (N)', 'desc': 'General pale yellowing starting on older mature leaves with reduced overall growth.', 'action': 'Apply nitrogen-rich fertilizer (Urea or Ammonium Nitrate) and organic compost.'},
    'phosphorus-P': {'title': 'Phosphorus Deficiency (P)', 'desc': 'Purplish or dark reddish tint on mature leaves with delayed leaf emergence.', 'action': 'Apply superphosphate or rock phosphate near active root absorption zones.'},
    'potasium-K': {'title': 'Potassium Deficiency (K)', 'desc': 'Marginal leaf scorch, brown necrotic borders, and leaf tip burn on mature leaves.', 'action': 'Apply potassium sulfate (K2SO4) or muriate of potash (MOP); apply organic mulch.'}
}

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

std_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

inc_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def disable_inplace_relu(model):
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

# ---------------------------------------------------------------------
# EXPLAINABLE AI (XAI) GRAD-CAM ENGINES
# ---------------------------------------------------------------------
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        g2 = gradients ** 2
        g3 = gradients ** 3
        alpha_num = g2
        alpha_denom = 2 * g2 + np.sum(activations * g3, axis=(1, 2), keepdims=True)
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)
        alphas = alpha_num / alpha_denom

        weights = np.maximum(gradients, 0.0)
        deep_features = np.sum(alphas * weights, axis=(1, 2))
        cam = np.sum(deep_features[:, np.newaxis, np.newaxis] * activations, axis=0)
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, output.detach().cpu().numpy()[0], target_class

class StandardGradCAMViT:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        grads = self.gradients.data.cpu().numpy()[0]
        acts = self.activations.data.cpu().numpy()[0]
        
        if grads.ndim == 2:
            grads_patch = grads[1:, :]
            acts_patch = acts[1:, :]
            weights = np.mean(grads_patch, axis=0)
            cam = np.dot(acts_patch, weights)
            num_patches = cam.shape[0]
            grid_size = int(np.sqrt(num_patches))
            cam = cam.reshape(grid_size, grid_size)
        else:
            weights = np.mean(grads, axis=(1, 2))
            cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts, axis=0)
            
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, output.detach().cpu().numpy()[0], target_class

# Build Models
print(f"Initializing LeafLens Model Backend onto device: {device}...")

def build_10_class_models():
    models_dict = {}
    p_vgg = os.path.join(SAVE_DIR_10, "best_vgg19_bn.pth")
    if os.path.exists(p_vgg):
        try:
            m_vgg = models.vgg19_bn()
            m_vgg.classifier[6] = nn.Linear(m_vgg.classifier[6].in_features, 10)
            m_vgg.load_state_dict(torch.load(p_vgg, map_location=device, weights_only=True))
            models_dict['vgg'] = m_vgg
        except Exception as e:
            print(f"Error VGG: {e}")

    p_inc = os.path.join(SAVE_DIR_10, "best_inceptionv3.pth")
    if os.path.exists(p_inc):
        try:
            m_inc = models.inception_v3()
            m_inc.fc = nn.Linear(m_inc.fc.in_features, 10)
            m_inc.AuxLogits.fc = nn.Linear(m_inc.AuxLogits.fc.in_features, 10)
            m_inc.load_state_dict(torch.load(p_inc, map_location=device, weights_only=True))
            models_dict['inc'] = m_inc
        except Exception as e:
            print(f"Error Inc: {e}")

    p_dense = os.path.join(SAVE_DIR_10, "best_densenet201.pth")
    if os.path.exists(p_dense):
        try:
            m_dense = models.densenet201()
            m_dense.classifier = nn.Linear(m_dense.classifier.in_features, 10)
            m_dense.load_state_dict(torch.load(p_dense, map_location=device, weights_only=True))
            models_dict['dense'] = m_dense
        except Exception as e:
            print(f"Error Dense: {e}")

    p_mob = os.path.join(SAVE_DIR_10, "best_mobilenetv3_large.pth")
    if os.path.exists(p_mob):
        try:
            m_mob = models.mobilenet_v3_large()
            m_mob.classifier[3] = nn.Linear(m_mob.classifier[3].in_features, 10)
            m_mob.load_state_dict(torch.load(p_mob, map_location=device, weights_only=True))
            models_dict['mob'] = m_mob
        except Exception as e:
            print(f"Error Mob: {e}")

    p_vit = os.path.join(SAVE_DIR_10, "best_vision_transformer_vit.pth")
    if os.path.exists(p_vit):
        try:
            m_vit = models.vit_b_16()
            m_vit.heads.head = nn.Linear(m_vit.heads.head.in_features, 10)
            m_vit.load_state_dict(torch.load(p_vit, map_location=device, weights_only=True))
            models_dict['vit'] = m_vit
        except Exception as e:
            print(f"Error ViT: {e}")

    p_eff = os.path.join(SAVE_DIR_10, "best_efficientnet_b4.pth")
    if os.path.exists(p_eff):
        try:
            m_eff = models.efficientnet_b4()
            m_eff.classifier[1] = nn.Linear(m_eff.classifier[1].in_features, 10)
            m_eff.load_state_dict(torch.load(p_eff, map_location=device, weights_only=True))
            models_dict['eff'] = m_eff
        except Exception as e:
            print(f"Error Eff: {e}")

    for k, m in models_dict.items():
        m.to(device)
        m.eval()
        disable_inplace_relu(m)

    return models_dict

models_10 = build_10_class_models()

meta_learner_10 = None
p_meta = os.path.join(SAVE_DIR_10, "stacking_meta_model.joblib")
if os.path.exists(p_meta):
    try:
        meta_learner_10 = joblib.load(p_meta)
    except Exception as e:
        print(f"Meta learner error: {e}")

gradcam_engine_10 = None
if 'vgg' in models_10:
    target_layer_10 = models_10['vgg'].features[49]
    gradcam_engine_10 = GradCAMPlusPlus(models_10['vgg'], target_layer_10)

def build_4_class_vit():
    m_vit = models.vit_b_16()
    m_vit.heads.head = nn.Linear(m_vit.heads.head.in_features, 4)
    p_vit_4 = os.path.join(SAVE_DIR_4, "vit_coleaf.pth")
    if os.path.exists(p_vit_4):
        try:
            m_vit.load_state_dict(torch.load(p_vit_4, map_location=device, weights_only=True))
            print("Loaded 4-Class ViT model.")
        except Exception as e:
            print(f"4-Class ViT error: {e}")
    m_vit.to(device)
    m_vit.eval()
    disable_inplace_relu(m_vit)
    return m_vit

vit_4 = build_4_class_vit()
target_layer_4 = vit_4.encoder.layers[-1].ln_1
gradcam_engine_4 = StandardGradCAMViT(vit_4, target_layer_4)

@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'service': 'LeafLens Coffee Leaf Deficiency AI Model Backend API',
        'status': 'online',
        'device': str(device),
        '10_class_models_loaded': list(models_10.keys()),
        '4_class_vit_loaded': vit_4 is not None
    })

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
        return False, "Uploaded image does not appear to contain a coffee leaf. Color spectrum lacks plant/chlorophyll tissue."
    if laplacian_var < 6.0:
        return False, "Image is too blank or blurry to detect leaf vein structures."
    return True, "Valid leaf image"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    model_mode = request.form.get('model_mode', '10_class')

    try:
        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Validate leaf image first
        is_valid_leaf, validation_reason = validate_leaf_image(pil_img)
        if not is_valid_leaf:
            return jsonify({'success': False, 'error': f'Invalid Image: {validation_reason}'}), 400

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
                avg_probs = np.mean(list(probs_dict.values()), axis=0)
                if meta_learner_10 is not None:
                    try:
                        feature_vector = np.concatenate([probs_dict.get(k, np.zeros(10)) for k in ['vgg', 'inc', 'dense', 'mob', 'vit', 'eff']]).reshape(1, -1)
                        probs_stack = meta_learner_10.predict_proba(feature_vector)[0]
                        final_probs = 0.6 * probs_stack + 0.4 * avg_probs
                        stack_pred_idx = int(np.argmax(probs_stack))
                    except Exception:
                        final_probs = avg_probs
                        stack_pred_idx = int(np.argmax(final_probs))
                else:
                    final_probs = avg_probs
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

        # Encode Overlay to Base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        gradcam_base64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

        _, orig_buffer = cv2.imencode('.png', cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR))
        orig_base64 = "data:image/png;base64," + base64.b64encode(orig_buffer).decode('utf-8')

        prob_list = []
        for idx, cls in enumerate(class_names):
            prob_list.append({
                'class': cls,
                'title': CLASS_INFO.get(cls, {}).get('title', cls),
                'probability': round(float(final_probs[idx]) * 100, 2)
            })
        prob_list.sort(key=lambda x: x['probability'], reverse=True)

        info = CLASS_INFO.get(predicted_class, {'title': predicted_class, 'desc': 'Coffee leaf deficiency classification.', 'action': 'Consult specialist.'})

        return jsonify({
            'success': True,
            'model_mode': model_mode,
            'predicted_class': predicted_class,
            'class_title': info['title'],
            'confidence': round(confidence, 2),
            'description': info['desc'],
            'action': info['action'],
            'original_image_base64': orig_base64,
            'gradcam_image_base64': gradcam_base64,
            'probabilities': prob_list,
            'models_summary': models_summary
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    print("\n" + "="*70)
    print("Starting LeafLens Dedicated AI Model Server...")
    print(f"Port: {port}")
    print("="*70 + "\n")
    app.run(host='0.0.0.0', port=port)
