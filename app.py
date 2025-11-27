import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from PIL import Image

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Face Recognition Cerdas",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓  Face Recognition Cerdas")
st.markdown("### Kelompok: Martua, Rayhan, Fadil")
st.markdown("---")

# ==========================================
# 2. LOAD MODEL & DATABASE (Di-Cache biar cepat)
# ==========================================
@st.cache_resource
def load_insightface_model():
    print("⏳ Memuat Model InsightFace...")
    # ctx_id=0 untuk GPU, ctx_id=-1 untuk CPU (Biar aman di laptop biasa pake -1)
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app

@st.cache_data
def load_database(path):
    print("📂 Memuat Database Wajah...")
    try:
        db = np.load(path, allow_pickle=True).item()
        return db
    except Exception as e:
        st.error(f"Gagal memuat database! Pastikan file {path} ada.")
        return {}

# Load resource
with st.spinner("Sedang menyiapkan sistem AI..."):
    app = load_insightface_model()
    face_db = load_database('insightface_db.npy')

# ==========================================
# 3. SIDEBAR (PENGATURAN)
# ==========================================
st.sidebar.header("⚙️ Pengaturan")
threshold = st.sidebar.slider("Akurasi Min (Threshold)", 0.0, 1.0, 0.50, 0.05)
st.sidebar.info(f"Jumlah Mahasiswa Terdaftar: **{len(face_db)}**")

# ==========================================
# 4. FUNGSI DETEKSI & PENGENALAN
# ==========================================
def process_image(image_file, mirror=False):
    # Baca gambar dari Streamlit (Bytes) -> OpenCV
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Mirror kamera SEBELUM deteksi (penting!)
    if mirror:
        img_rgb = cv2.flip(img_rgb, 1)  # 1 = flip horizontal
        img_bgr = cv2.flip(img_bgr, 1)
    
    print(f"🔍 Mirror aktif: {mirror}")  # Debug
    
    # Deteksi Wajah
    faces = app.get(img_bgr)
    
    # Siapkan Canvas untuk menggambar
    img_draw = img_rgb.copy()
    
    results = []
    
    if len(faces) == 0:
        st.warning("⚠️ Tidak ada wajah yang terdeteksi.")
        return img_draw, results

    # Loop setiap wajah yang ketemu
    for face in faces:
        bbox = face.bbox.astype(int)
        emb = face.normed_embedding
        
        # --- PROSES PENCOCOKAN (RECOGNITION) ---
        max_score = -1
        best_name = "Unknown"
        
        for name, db_emb in face_db.items():
            # Hitung kemiripan (Cosine Similarity)
            score = np.dot(emb, db_emb)
            if score > max_score:
                max_score = score
                best_name = name
        
        # Cek Threshold
        final_name = "Tidak Dikenal"
        color = (255, 0, 0) # Merah (Unknown)
        
        if max_score >= threshold:
            final_name = best_name
            color = (0, 255, 0) # Hijau (Dikenal)
            results.append(final_name)
        
        # Gambar Kotak & Nama
        cv2.rectangle(img_draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        cv2.putText(img_draw, f"{final_name} ({max_score:.2f})", 
                    (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return img_draw, results

# ==========================================
# 5. UI UTAMA (TABS)
# ==========================================
tab1, tab2 = st.tabs(["📸 Ambil Foto (Live)", "📂 Upload File"])

with tab1:
    st.write("Ambil foto selfie untuk presensi:")
    mirror_cam = st.checkbox("🔄 Mirror Kamera", value=True)
    camera_img = st.camera_input("Kamera")
    
    if camera_img is not None:
        processed_img, names = process_image(camera_img, mirror=mirror_cam)
        st.image(processed_img, caption="Hasil Deteksi", use_container_width=True)
        
        if names:
            st.success(f"✅ Presensi Berhasil: **{', '.join(names)}**")
        else:
            st.error("❌ Wajah tidak terdaftar.")

with tab2:
    uploaded_file = st.file_uploader("Upload foto (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        processed_img, names = process_image(uploaded_file, mirror=False)
        st.image(processed_img, caption="Hasil Analisis", use_container_width=True)
        
        if names:
            st.success(f"✅ Teridentifikasi: **{', '.join(names)}**")
        else:
            st.error("❌ Tidak ada mahasiswa yang dikenali.")