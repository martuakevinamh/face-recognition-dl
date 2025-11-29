import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download
from facenet_pytorch import MTCNN 

# ==========================================
# 1. KONFIGURASI
# ==========================================
st.set_page_config(page_title="Presensi ViT Smart Crop", page_icon="🧠", layout="centered")
HF_REPO_ID = "Martua/tubes-deep-learning-face"  

st.title("🧠 Face Recognition (ViT)")
st.caption("Menggunakan MTCNN untuk deteksi wajah & ViT untuk mengenali identitas.")
st.markdown("---")

# ==========================================
# 2. LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_face_detector():
    # MTCNN adalah detektor wajah standar industri yang ringan
    return MTCNN(keep_all=False, select_largest=True, device='cpu')

@st.cache_resource
def get_file_from_hf(filename):
    try:
        return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    except:
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
    device = torch.device("cpu")
    try:
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.heads.head.in_features, num_classes)
        )
        path = get_file_from_hf("model_vit_augmented_martua.pth")
        if path:
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            return model
    except:
        pass
    return None

# Init
with st.spinner("Menyiapkan AI (Detektor + Pengenal)..."):
    mtcnn = load_face_detector()
    labels = load_labels()
    model = load_vit_model(len(labels)) if labels else None

# ==========================================
# 3. FUNGSI PREDIKSI CERDAS
# ==========================================
def predict_smart(img_pil, threshold=0.6):
    if model is None: return "Error Model", 0.0, img_pil
    
    # --- TAHAP 1: DETEKSI & CROP WAJAH ---
    # MTCNN akan otomatis mencari wajah dan memotongnya
    # Outputnya langsung tensor atau PIL Image yang sudah di-crop
    try:
        face_tensor = mtcnn(img_pil) # Deteksi wajah
    except:
        face_tensor = None

    if face_tensor is None:
        return "Wajah Tidak Ditemukan", 0.0, img_pil
    
    # Konversi tensor balik ke PIL Image untuk ditampilkan ke user (Visualisasi Crop)
    face_img_viz = transforms.ToPILImage()(face_tensor)

    # --- TAHAP 2: PREDIKSI ViT ---
    # Transformasi khusus ViT
    transform_vit = transforms.Compose([
        transforms.Resize((224, 224)), # Pastikan ukuran pas buat ViT
        # Tidak perlu ToTensor lagi karena output MTCNN sudah tensor (jika dikonfig begitu)
        # Tapi biar aman kita normalize manual dari tensor MTCNN
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # MTCNN outputnya range 0-1 atau -1..1 tergantung setting, 
    # kita pastikan formatnya sesuai input ViT
    # Cara aman: face_img_viz -> Transform ViT Standard
    
    transform_standard = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform_standard(face_img_viz).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
    
    confidence = conf.item()
    pred_name = labels[idx.item()]
    
    if confidence > threshold:
        return pred_name, confidence, face_img_viz
    else:
        return f"Wajah Asing ({pred_name})", confidence, face_img_viz

# ==========================================
# 4. UI APLIKASI
# ==========================================
col_conf, col_stat = st.columns(2)
with col_conf:
    thresh = st.slider("Threshold Keyakinan", 0.0, 1.0, 0.60)
with col_stat:
    st.info(f"Database: **{len(labels)} Mahasiswa**")

mode = st.radio("Input:", ["📸 Kamera", "📂 Upload"], horizontal=True)
img_file = st.camera_input("Foto") if mode == "📸 Kamera" else st.file_uploader("Upload", type=['jpg','png','jpeg'])

if img_file:
    img_pil = Image.open(img_file).convert("RGB")
    
    # Layout 2 Kolom: Asli vs Hasil Crop
    c1, c2 = st.columns(2)
    
    with c1:
        st.image(img_pil, caption="Foto Asli (Full)", use_container_width=True)
        
    with c2:
        with st.spinner("Mendeteksi & Mengenali Wajah..."):
            name, score, cropped_face = predict_smart(img_pil, thresh)
        
        # Tampilkan Wajah yang Dideteksi (Crop)
        st.image(cropped_face, caption="Wajah Terdeteksi (Input ke ViT)", width=200)
        
        # Hasil
        if "Tidak Ditemukan" in name:
            st.warning("⚠️ Wajah tidak terdeteksi oleh MTCNN.")
        elif "Asing" in name:
            st.error(f"❌ **{name}**")
            st.caption("Sistem ragu-ragu (Score rendah).")
        elif "Error" in name:
            st.error("Gagal memuat model.")
        else:
            st.success(f"✅ **{name}**")
            st.progress(score, text=f"Confidence: {score*100:.1f}%")
            
            # Penjelasan Logis buat Demo
            st.info("💡 **Info Teknis:** Sistem berhasil memotong bagian wajah dan mengabaikan background/baju, sehingga prediksi lebih akurat.")