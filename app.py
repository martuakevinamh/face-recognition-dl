import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
from huggingface_hub import hf_hub_download
import os

# ==========================================
# 1. KONFIGURASI HALAMAN & REPO
# ==========================================
st.set_page_config(
    page_title="Face Recognition Cerdas",
    page_icon="🎓",
    layout="wide"
)

HF_REPO_ID = "Martua/tubes-deep-learning-face" 

st.title("🎓 Face Recognition Cerdas ")
st.markdown("### Kelompok: Martua, Rayhan, Fadil")
st.markdown("---")

# ==========================================
# 2. FUNGSI DOWNLOAD DARI HUGGING FACE
# ==========================================
@st.cache_resource
def get_file_from_hf(filename):
    try:
        # Ini akan download file dan menyimpannya di cache lokal
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
        return path
    except Exception as e:
        st.error(f"Gagal download {filename} dari Hugging Face: {e}")
        return None

# ==========================================
# 3. LOAD MODEL INSIGHTFACE (ZERO-SHOT)
# ==========================================
@st.cache_resource
def load_insightface():
    print("⏳ Memuat InsightFace...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app

@st.cache_data
def load_insightface_db():
    # Download DB dulu kalau belum ada
    db_path = get_file_from_hf("insightface_db.npy")
    if db_path:
        return np.load(db_path, allow_pickle=True).item()
    return {}

# ==========================================
# 4. LOAD MODEL ViT (FINE-TUNED)
# ==========================================
@st.cache_resource
def load_vit_model(num_classes):
    print("⏳ Memuat ViT...")
    device = torch.device("cpu")
    
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.heads.head.in_features, num_classes)
    )
    
    # Download Model ViT dari HF
    model_path = get_file_from_hf("model_vit_tuned_martua.pth")
    
    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    return None

@st.cache_data
def load_labels():
    # Download Labels dari HF
    label_path = get_file_from_hf("labels_pytorch.txt")
    if label_path:
        with open(label_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return []

# --- EKSEKUSI LOAD RESOURCE ---
with st.spinner("Sedang mengunduh model dari Hugging Face..."):
    app_insight = load_insightface()
    db_insight = load_insightface_db()
    
    labels_vit = load_labels()
    # Pastikan labels terload sebelum load model
    if labels_vit:
        model_vit = load_vit_model(len(labels_vit))
    else:
        model_vit = None

# ==========================================
# 5. FUNGSI PREDIKSI 
# ==========================================
def predict_insightface(img_bgr, threshold=0.5):
    faces = app_insight.get(img_bgr)
    if len(faces) == 0: return "Wajah Tidak Terdeteksi", 0.0
    
    face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)[0]
    emb = face.normed_embedding
    
    max_score = 0
    best_name = "Unknown"
    for name, db_emb in db_insight.items():
        score = np.dot(emb, db_emb)
        if score > max_score:
            max_score = score
            best_name = name
            
    return (best_name, float(max_score)) if max_score > threshold else ("Tidak Dikenal", float(max_score))

def predict_vit(img_pil, threshold=0.5):
    if model_vit is None: return "Model Error", 0.0
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        outputs = model_vit(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
    return (labels_vit[idx.item()], conf.item()) if conf.item() > threshold else ("Tidak Dikenal", conf.item())

# ==========================================
# 6. UI UTAMA
# ==========================================
col1, col2 = st.columns(2)
with col1:
    thresh_insight = st.slider("Threshold InsightFace", 0.0, 1.0, 0.5)
with col2:
    thresh_vit = st.slider("Threshold ViT", 0.0, 1.0, 0.5)

col3, col4 = st.columns(2)
with col3:
    mirror_image = st.toggle("🔄 Mirror Gambar", value=False)
with col4:
    st.empty()  # Placeholder untuk balance layout

input_mode = st.radio("Mode:", ["📸 Kamera", "📂 Upload File"], horizontal=True)
image_input = st.camera_input("Foto") if input_mode == "📸 Kamera" else st.file_uploader("Upload", type=['jpg','png','jpeg'])

if image_input:
    file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Apply mirror jika toggle aktif
    if mirror_image:
        img_bgr = cv2.flip(img_bgr, 1)
        img_rgb = cv2.flip(img_rgb, 1)
    
    image_input.seek(0)
    img_pil = Image.open(image_input).convert("RGB")
    
    # Apply mirror pada PIL image jika toggle aktif
    if mirror_image:
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    
    st.image(img_rgb, width=400)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("🤖 InsightFace")
        name, score = predict_insightface(img_bgr, thresh_insight)
        if name != "Tidak Dikenal":
            st.success(f"**{name}**")
        else:
            st.error(name)
        st.progress(min(score, 1.0), f"{score:.2f}")
        
    with c2:
        st.warning("🧠 ViT (Fine-Tuned)")
        name, score = predict_vit(img_pil, thresh_vit)
        if name != "Tidak Dikenal":
            st.success(f"**{name}**")
        else:
            st.error(name)
        st.progress(min(score, 1.0), f"{score:.2f}")