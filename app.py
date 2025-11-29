import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download
from facenet_pytorch import MTCNN
import os

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Sistem Presensi Cerdas",
    page_icon="🎓",
    layout="centered"
)

# ⚠️ GANTI DENGAN USERNAME/NAMA_MODEL KAMU DI HUGGING FACE
HF_REPO_ID = "Martua/tubes-deep-learning-face" 

st.title("🎓 Presensi Mahasiswa Berbasis AI")
st.caption("Didukung oleh Custom Vision Transformer (ViT-B/16) & MTCNN")
st.markdown("---")

# ==========================================
# 2. HELPER DOWNLOAD & LOAD MODEL
# ==========================================
@st.cache_resource
def get_file_from_hf(filename):
    try:
        return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    except Exception as e:
        st.error(f"Gagal mengunduh {filename} dari Hugging Face: {e}")
        return None

@st.cache_data
def load_labels():
    path = get_file_from_hf("labels_augmented.txt")
    if path:
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return []

@st.cache_resource
def load_vit_model(num_classes):
    print("⏳ Memuat Model ViT...")
    device = torch.device("cpu") # Aman untuk Cloud/Laptop tanpa GPU
    
    try:
        # 1. Load Arsitektur ViT Kosong
        model = models.vit_b_16(weights=None)
        
        # 2. Sesuaikan Kepala Model (Sesuai Training 100% kemarin)
        model.heads.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.heads.head.in_features, num_classes)
        )
        
        # 3. Load Bobot dari Hugging Face
        model_path = get_file_from_hf("model_vit_augmented_martua.pth")
        
        if model_path:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval() # Mode Evaluasi
            return model
            
    except Exception as e:
        st.error(f"Error memuat model ViT: {e}")
    return None

@st.cache_resource
def load_face_detector():
    # MTCNN untuk deteksi wajah
    return MTCNN(keep_all=False, select_largest=True, device='cpu')

# --- INISIALISASI SISTEM ---
with st.spinner("Sedang menyiapkan kecerdasan buatan..."):
    mtcnn = load_face_detector()
    labels = load_labels()
    
    if labels:
        model = load_vit_model(len(labels))
    else:
        model = None
        st.error("Gagal memuat label nama.")

# ==========================================
# 3. FUNGSI PREDIKSI CERDAS (MTCNN -> ViT)
# ==========================================
def predict_smart(img_pil, threshold=0.6):
    if model is None: return "Error Model", 0.0, None
    
    # TAHAP 1: DETEKSI & CROP WAJAH
    try:
        # MTCNN mengembalikan tensor wajah yang sudah di-crop
        face_tensor = mtcnn(img_pil) 
    except:
        face_tensor = None

    if face_tensor is None:
        return "Wajah Tidak Ditemukan", 0.0, None
    
    # Konversi tensor balik ke PIL Image untuk visualisasi crop di UI
    face_img_viz = transforms.ToPILImage()(face_tensor)

    # TAHAP 2: PREDIKSI ViT
    # Transformasi standar ViT (Resize 224, Normalize)
    transform_vit = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Siapkan input batch
    input_tensor = transform_vit(face_img_viz).unsqueeze(0)

    # Prediksi
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
    
    confidence = conf.item()
    pred_name = labels[idx.item()]
    
    # Logika Threshold
    if confidence > threshold:
        return pred_name, confidence, face_img_viz
    else:
        return f"Wajah Asing ({pred_name}?)", confidence, face_img_viz

# ==========================================
# 4. USER INTERFACE
# ==========================================
# Sidebar Kontrol
st.sidebar.header("⚙️ Panel Kontrol")
threshold = st.sidebar.slider("Tingkat Keyakinan (Threshold)", 0.0, 1.0, 0.60)
st.sidebar.info(f"Database: **{len(labels)} Mahasiswa**")
st.sidebar.markdown("---")
st.sidebar.caption("Kelompok: Martua, Rayhan, Fadil")

# Pilihan Input
mode = st.radio("Metode Input:", ["📸 Ambil Foto (Live)", "📂 Upload File"], horizontal=True)

image_input = None
if mode == "📸 Ambil Foto (Live)":
    image_input = st.camera_input("Silakan ambil foto wajah")
else:
    image_input = st.file_uploader("Upload foto wajah (JPG/PNG)", type=['jpg','png','jpeg'])

# Proses Gambar
if image_input:
    # Buka gambar
    img_pil = Image.open(image_input).convert("RGB")
    
    # Layout Kolom (Input vs Output)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img_pil, caption="Foto Asli", use_container_width=True)
    
    with col2:
        st.markdown("### Hasil Analisis")
        
        with st.spinner("Mendeteksi & Mengenali..."):
            name, score, cropped_face = predict_smart(img_pil, threshold)
        
        # Tampilkan Hasil
        if "Tidak Ditemukan" in name:
            st.warning("⚠️ **Wajah tidak terdeteksi.**")
            st.caption("Coba posisikan wajah lebih jelas.")
            
        elif "Asing" in name:
            st.error(f"❌ **{name}**")
            st.progress(score, text=f"Confidence: {score*100:.1f}% (Rendah)")
            if cropped_face:
                st.image(cropped_face, caption="Input ke Model", width=120)
                
        elif "Error" in name:
            st.error("Gagal memuat model AI.")
            
        else:
            st.success(f"✅ **Teridentifikasi: {name}**")
            st.progress(score, text=f"Confidence: {score*100:.1f}%")
            
            if score > 0.9:
                st.balloons()
            
            if cropped_face:
                st.image(cropped_face, caption="Wajah Terdeteksi", width=120)
                st.info("💡 Sistem berhasil memotong wajah & mengabaikan background.")