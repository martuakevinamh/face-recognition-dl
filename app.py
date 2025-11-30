import streamlit as st
import torch
import timm
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from huggingface_hub import hf_hub_download
from torchvision import transforms

# ==========================================
# 1. KONFIGURASI
# ==========================================
st.set_page_config(
    page_title="Sistem Presensi ViT",
    page_icon="👁️",
    layout="centered"
)

# GANTI DENGAN REPO KAMU
HF_REPO_ID = "Martua/tubes-deeplearning"
MODEL_FILENAME = "face_vit_svm_augmented.pth"

st.title("👁️ Sistem Presensi Wajah")
st.caption("🚀 Architecture: ViT-Base (Feature Extractor) + SVM (Classifier)")
st.markdown("---")

# ==========================================
# 2. LOAD ENGINE (CACHE BIAR NGEBUT)
# ==========================================
@st.cache_resource
def get_model_path(filename):
    try:
        return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    except Exception as e:
        st.error(f"Gagal download model: {e}")
        return None

@st.cache_resource
def load_engine():
    print("⏳ Memuat ViT Engine...")
    device = torch.device('cpu')
    
    # 1. MTCNN (Detektor)
    # Penting: Image size harus 224 sesuai input ViT
    mtcnn = MTCNN(
        image_size=224, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    
    # 2. ViT (Feature Extractor)
    # Kita pakai model yang sama persis dengan training
    vit = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=False, num_classes=0)
    vit.eval()
    return mtcnn, vit

@st.cache_resource
def load_svm_model():
    print("⏳ Memuat Otak SVM...")
    path = get_model_path(MODEL_FILENAME)
    
    if path:
        try:
            # Load file .pth (isi: classifier & classes)
            state = torch.load(path, map_location='cpu', weights_only=False)
            return state['classifier'], state['classes']
        except Exception as e:
            st.error(f"Error load SVM: {e}")
    return None, None

# Init System
with st.spinner("Sedang menyiapkan kecerdasan buatan..."):
    mtcnn, vit = load_engine()
    clf, class_names = load_svm_model()
    
    if clf:
        st.success(f"✅ Sistem Siap! Database: **{len(class_names)} Mahasiswa**")
    else:
        st.stop()

# ==========================================
# 3. FUNGSI PREDIKSI
# ==========================================
def predict(img_pil, threshold=0.5):
    # 1. Deteksi & Crop Wajah
    # MTCNN otomatis resize ke 224x224
    img_cropped, prob = mtcnn(img_pil, return_prob=True)
    
    if img_cropped is not None and prob > 0.90:
        # 2. Preprocessing (Normalisasi ala ViT)
        # MTCNN output range 0..1 -> Kita ubah jadi standar ViT
        # Rumus: (image - 0.5) / 0.5
        face_input = (img_cropped - 0.5) / 0.5 
        face_input = face_input.unsqueeze(0) # Tambah batch dimension
        
        # 3. Ekstrak Fitur (ViT)
        with torch.no_grad():
            emb = vit(face_input).numpy()
            
        # 4. Prediksi (SVM)
        probs = clf.predict_proba(emb)
        max_prob = np.max(probs)
        name = clf.predict(emb)[0]
        
        # Visualisasi Wajah Crop (Balikin ke range 0-255 biar bisa dilihat)
        face_viz = transforms.ToPILImage()(img_cropped)
        
        # 5. Thresholding
        if max_prob > threshold:
            return name, max_prob, face_viz
        else:
            return f"Wajah Asing ({name}?)", max_prob, face_viz
            
    return "Wajah Tidak Terdeteksi", 0.0, None

# ==========================================
# 4. USER INTERFACE
# ==========================================
threshold = st.slider("Sensitivitas (Threshold)", 0.0, 1.0, 0.50)
mode = st.radio("Input:", ["📸 Kamera", "📂 Upload"], horizontal=True)

img_file = st.camera_input("Foto") if mode == "📸 Kamera" else st.file_uploader("Upload", type=['jpg','png','jpeg'])

if img_file:
    img = Image.open(img_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Foto Input", use_container_width=True)
    
    with col2:
        st.markdown("### Hasil Analisis")
        with st.spinner("Menganalisis biometrik..."):
            name, conf, face_crop = predict(img, threshold)
        
        if "Tidak" in name or "Wajah" in name:
            st.warning(f"⚠️ **{name}**")
            if face_crop:
                st.image(face_crop, caption="Wajah Terdeteksi", width=120)
        else:
            st.success(f"✅ **Teridentifikasi: {name}**")
            st.progress(conf, text=f"Confidence: {conf*100:.1f}%")
            if face_crop:
                st.image(face_crop, caption="Wajah Terdeteksi", width=120)
            if conf > 0.8:
                st.balloons()