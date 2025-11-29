import streamlit as st
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from huggingface_hub import hf_hub_download
import cv2

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Presensi Wajah SVM",
    page_icon="👤",
    layout="centered"
)

# ⚠️ SESUAIKAN DENGAN REPO KAMU (Huruf Besar/Kecil Pengaruh!)
HF_REPO_ID = "Martua/tubes-deep-learning-face"
MODEL_FILENAME = "face_svm_augmented.pth"

st.title("👤 Sistem Presensi Wajah")
st.caption("🚀 Powered by: InceptionResnetV1 (Feature) + SVM (Classifier)")
st.markdown("---")

# ==========================================
# 2. HELPER DOWNLOAD & LOAD
# ==========================================
@st.cache_resource
def get_model_path(filename):
    try:
        return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    except Exception as e:
        st.error(f"Gagal download {filename}: {e}")
        return None

@st.cache_resource
def load_face_engine():
    print("⏳ Memuat Facenet Engine...")
    device = torch.device('cpu') 
    
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
    model_path = get_model_path(MODEL_FILENAME)
    
    if model_path:
        try:
            # Load file .pth yang berisi dictionary {'classifier': ..., 'classes': ...}
            state = torch.load(model_path, map_location='cpu')
            return state['classifier'], state['classes']
        except Exception as e:
            st.error(f"Error membaca model SVM: {e}")
    return None, None

# --- INIT ---
with st.spinner("Sedang menyiapkan sistem..."):
    mtcnn, resnet = load_face_engine()
    clf, class_names = load_svm_model()

    if clf:
        st.success("✅ Sistem Siap!")
    else:
        st.stop() # Berhenti jika model gagal load

# ==========================================
# 3. FUNGSI PREDIKSI
# ==========================================
def predict_face(img_pil, threshold=0.5):
    # 1. Deteksi & Crop
    try:
        img_cropped, prob = mtcnn(img_pil, return_prob=True)
    except:
        return "Error Deteksi", 0.0, None

    if img_cropped is not None and prob > 0.90:
        # 2. Ekstrak Fitur (Embedding)
        with torch.no_grad():
            img_embedding = resnet(img_cropped.unsqueeze(0)) # Tambah batch dimension
        
        embedding_np = img_embedding.detach().numpy()
        
        # 3. Prediksi Nama pakai SVM
        prediction = clf.predict(embedding_np)
        probabilities = clf.predict_proba(embedding_np)
        
        max_prob = np.max(probabilities)
        name = prediction[0]
        
        # Visualisasi crop wajah
        face_viz = transforms.ToPILImage()(img_cropped)

        # 4. Filter Keyakinan
        if max_prob > threshold:
            return name, max_prob, face_viz
        else:
            return f"Wajah Asing ({name}?)", max_prob, face_viz
            
    return "Wajah Tidak Terdeteksi", 0.0, None

# ==========================================
# 4. USER INTERFACE
# ==========================================
from torchvision import transforms

# Sidebar
threshold = st.sidebar.slider("Sensitivitas (Threshold)", 0.0, 1.0, 0.50)
if class_names is not None:
    st.sidebar.info(f"Database: **{len(class_names)} Mahasiswa**")

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
        st.markdown("### Hasil Analisis")
        
        with st.spinner("Mengidentifikasi..."):
            name, conf, face_crop = predict_face(img_pil, threshold)
        
        # Tampilkan Hasil
        if "Tidak" in name or "Error" in name:
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