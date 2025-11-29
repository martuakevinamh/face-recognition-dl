import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import os

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Presensi Wajah Berbasis ViT",
    page_icon="🎓",
    layout="centered" # Fokus di tengah biar elegan
)

# ⚠️ JANGAN LUPA GANTI INI
HF_REPO_ID = "Martua/tubes-deep-learning-face" 

st.title("🎓 Sistem Presensi Cerdas")
st.caption("Powered by Custom Vision Transformer (ViT-B/16)")
st.markdown("---")

# ==========================================
# 2. HELPER DOWNLOAD & LOAD MODEL
# ==========================================
@st.cache_resource
def get_file_from_hf(filename):
    try:
        return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    except Exception as e:
        st.error(f"Gagal download {filename}: {e}")
        return None

@st.cache_data
def load_labels():
    path = get_file_from_hf("labels_augmented.txt")
    if path:
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return []

@st.cache_resource
def load_model(num_classes):
    device = torch.device("cpu") # Aman untuk Cloud
    
    # Load Arsitektur ViT
    try:
        model = models.vit_b_16(weights=None)
        
        # Sesuaikan Head (Sesuai Training 100% kemarin)
        model.heads.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.heads.head.in_features, num_classes)
        )
        
        # Load Bobot
        model_path = get_file_from_hf("model_vit_augmented_martua.pth")
        if model_path:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            return model
    except Exception as e:
        st.error(f"Error Loading Model: {e}")
    return None

# --- INIT ---
with st.spinner("Sedang memuat model kecerdasan buatan..."):
    labels = load_labels()
    model = load_model(len(labels)) if labels else None

# ==========================================
# 3. FUNGSI PREDIKSI (Klasifikasi)
# ==========================================
def predict_face(img_pil, threshold=0.6):
    if model is None: return "Error Model", 0.0
    
    # Transformasi Standar ViT
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
        
    confidence = conf.item()
    pred_name = labels[idx.item()]
    
    if confidence > threshold:
        return pred_name, confidence
    else:
        return "Wajah Tidak Dikenal", confidence

# ==========================================
# 4. USER INTERFACE
# ==========================================
# Sidebar untuk tuning saat demo
st.sidebar.header("⚙️ Panel Kontrol")
threshold = st.sidebar.slider("Tingkat Keyakinan (Threshold)", 0.0, 1.0, 0.60)
st.sidebar.info(f"Model dilatih untuk mengenali **{len(labels)}** mahasiswa.")

# Input Mode
mode = st.radio("Pilih Metode Input:", ["📸 Ambil Foto (Live)", "📂 Upload File"], horizontal=True)

image_input = None
if mode == "📸 Ambil Foto (Live)":
    image_input = st.camera_input("Silakan ambil foto wajah")
else:
    image_input = st.file_uploader("Upload foto wajah (JPG/PNG)", type=['jpg','png','jpeg'])

# Proses jika ada gambar
if image_input:
    # Tampilkan Gambar
    img_pil = Image.open(image_input).convert("RGB")
    
    # Buat kolom biar rapi
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(img_pil, caption="Foto Input", use_container_width=True)
    
    with col2:
        st.markdown("### Hasil Analisis:")
        
        # Prediksi
        with st.spinner("Menganalisis fitur wajah..."):
            name, score = predict_face(img_pil, threshold)
        
        # Tampilan Hasil
        if name == "Wajah Tidak Dikenal":
            st.warning(f"⚠️ **{name}**")
            st.write("Kemiripan terlalu rendah.")
        elif name == "Error Model":
            st.error("Model gagal dimuat.")
        else:
            st.success(f"✅ **Teridentifikasi: {name}**")
            # Efek visual progress bar
            st.progress(score, text=f"Tingkat Keyakinan: {score*100:.1f}%")
            
            if score > 0.9:
                st.balloons() # Efek balon kalau yakin banget (Gimmick buat demo!)