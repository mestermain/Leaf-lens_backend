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

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES_4 = ['healthy', 'nitrogen-N', 'phosphorus-P', 'potasium-K']
CLASS_INFO = {
    'healthy': {'title': 'Healthy Leaf', 'desc': 'Vibrant green color, optimal photosynthetic activity.', 'action': 'Maintain standard irrigation.'},
    'nitrogen-N': {'title': 'Nitrogen Deficiency (N)', 'desc': 'General chlorosis of older leaves.', 'action': 'Apply nitrogen-rich fertilizer.'},
    'phosphorus-P': {'title': 'Phosphorus Deficiency (P)', 'desc': 'Dark green with purplish margins.', 'action': 'Apply phosphate fertilizers.'},
    'potasium-K': {'title': 'Potassium Deficiency (K)', 'desc': 'Yellowing and necrosis of leaf margins.', 'action': 'Apply potassium sulfate.'},
}

std_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

def build_4_class_vit():
    m_vit = models.vit_b_16(weights=None)
    m_vit.heads.head = nn.Linear(m_vit.heads.head.in_features, 4)
    p_vit = r"C:\Users\PC\Desktop\Four-Classes_Coleaf\Saved_Models\vit_coleaf.pth"
    if os.path.exists(p_vit):
        m_vit.load_state_dict(torch.load(p_vit, map_location=device))
    m_vit.to(device)
    m_vit.eval()
    return m_vit

vit_4 = build_4_class_vit()
target_layer_4 = vit_4.encoder.layers[-1].ln_1
gradcam_engine_4 = StandardGradCAMViT(vit_4, target_layer_4)

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

@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'service': 'LeafLens Coffee Leaf Deficiency AI Model Backend API',
        'status': 'online',
        'device': str(device),
        '4_class_vit_loaded': vit_4 is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    try:
        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        is_valid_leaf, validation_reason = validate_leaf_image(pil_img)
        if not is_valid_leaf:
            return jsonify({'success': False, 'error': f'Invalid Image: {validation_reason}'}), 400

        cv_img_rgb = np.array(pil_img)
        h, w, _ = cv_img_rgb.shape
        
        input_std = std_transform(pil_img).unsqueeze(0).to(device)

        class_names = CLASS_NAMES_4
        cam, logits, pred_idx = gradcam_engine_4.generate(input_std)
        probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
        predicted_class = class_names[pred_idx]
        confidence = float(probs[pred_idx]) * 100.0
        final_probs = probs
        models_summary = {'Vision Transformer (ViT)': {'class': predicted_class, 'confidence': round(confidence, 2)}}

        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(cv_img_rgb, 0.55, heatmap_rgb, 0.45, 0)

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
            'model_mode': '4_class',
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
