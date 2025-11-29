import streamlit as st
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Presensi Wajah Pro", page_icon="👤", layout="centered")

st.title("👤 Sistem Presensi Wajah")
st.caption("🚀 Model: InceptionResnetV1 (Feature) + SVM (Classifier) | Akurasi: 94%")
st.markdown("---")

# ==========================================
# 2. LOAD ENGINE AI (Di-Cache Biar Ngebut)
# ==========================================
@st.cache_resource
def load_face_engine():
    print("⏳ Memuat Engine AI...")
    device = torch.device('cpu') # Aman buat laptop
    
    # 1. Detektor Wajah (MTCNN)
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    
    # 2. Ekstraktor Fitur (InceptionResnetV1)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, resnet

@st.cache_resource
def load_svm_model():
    print("⏳ Memuat Otak SVM...")
    try:
        # Load file .pth hasil training kamu
        state = torch.load('face_svm_augmented.pth', map_location='cpu')
        return state['classifier']
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

# Inisialisasi
with st.spinner("Sedang menyiapkan sistem cerdas..."):
    mtcnn, resnet = load_face_engine()
    clf = load_svm_model()

# ==========================================
# 3. LOGIKA PREDIKSI
# ==========================================
def predict_face(img_pil, threshold=0.5):
    # 1. Deteksi & Crop Wajah
    img_cropped, prob = mtcnn(img_pil, return_prob=True)
    
    if img_cropped is not None and prob > 0.90:
        # 2. Ubah Wajah jadi Angka (Embedding)
        img_embedding = resnet(img_cropped.unsqueeze(0)) # Tambah dimensi batch
        embedding_np = img_embedding.detach().numpy()
        
        # 3. Prediksi Nama pakai SVM
        prediction = clf.predict(embedding_np)
        probability = clf.predict_proba(embedding_np)
        
        max_prob = np.max(probability)
        name = prediction[0]
        
        # 4. Cek Keyakinan (Threshold)
        if max_prob > threshold:
            return name, max_prob, img_cropped
        else:
            return "Wajah Tidak Dikenal", max_prob, img_cropped
            
    return "No Face", 0.0, None

# ==========================================
# 4. TAMPILAN (UI)
# ==========================================
# Sidebar
thresh = st.sidebar.slider("Sensitivitas (Threshold)", 0.0, 1.0, 0.50)
st.sidebar.info(f"Database: **{len(clf.classes_)} Mahasiswa**")

# Input
mode = st.radio("Metode Input:", ["📸 Kamera", "📂 Upload File"], horizontal=True)

image_input = None
if mode == "📸 Kamera":
    image_input = st.camera_input("Ambil Foto Presensi")
else:
    image_input = st.file_uploader("Upload Foto", type=['jpg','png','jpeg'])

# Eksekusi
if image_input:
    img_pil = Image.open(image_input).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img_pil, caption="Foto Asli", use_container_width=True)
    
    with col2:
        st.write("### Hasil Analisis")
        
        with st.spinner("Mengidentifikasi..."):
            name, conf, face_tensor = predict_face(img_pil, thresh)
        
        if name == "No Face":
            st.error("❌ Wajah tidak ditemukan.")
        elif name == "Wajah Tidak Dikenal":
            st.warning(f"⚠️ **{name}**")
            st.caption(f"Confidence: {conf*100:.1f}% (Kurang Yakin)")
            # Tampilkan wajah yang dideteksi
            if face_tensor is not None:
                # Convert tensor ke gambar buat ditampilkan
                face_img = face_tensor.permute(1, 2, 0).int().numpy()
                st.image(face_img, caption="Input Wajah", width=100)
        else:
            st.success(f"✅ **Halo, {name}!**")
            st.progress(conf, text=f"Akurasi: {conf*100:.1f}%")
            st.balloons()
            
            # Tampilkan wajah yang dideteksi
            if face_tensor is not None:
                face_img = face_tensor.permute(1, 2, 0).int().numpy()
                st.image(face_img, caption="Wajah Terdeteksi", width=120)