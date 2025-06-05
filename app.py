import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import io
from edukasi import waste_info

# HARUS DILETAKKAN PALING AWAL SEBELUM ST LAINNYA
st.set_page_config(page_title="WasteTrack", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_classifier_model.h5")

model = load_model()
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Sesuai dataset kamu

# Fungsi prediksi
def predict(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return predicted_class, confidence

# Navigasi
menu = st.sidebar.selectbox("Navigasi", ["Home", "Deteksi Sampah", "Jenis Sampah", "Tentang Kami"])

# Home
if menu == "Home":
    st.title("♻️ WasteTrack: Deteksi dan Edukasi Sampah")
    st.markdown("""
    Selamat datang di **WasteTrack**!  
    Aplikasi ini dirancang untuk membantu mengidentifikasi jenis sampah dan memberikan edukasi tentang bagaimana cara mengelolanya.  
    ### 🔍 Fitur:
    - Deteksi jenis sampah dari gambar.
    - Informasi apakah bisa didaur ulang.
    - Tips mengelola sampah organik & anorganik.

    ### 💡 Cara Menggunakan:
    1. Masuk ke menu **Deteksi Sampah**.
    2. Upload atau ambil gambar.
    3. Klik **Deteksi Sampah**.
    4. Lihat hasil & informasi edukatif.
    """)

# Deteksi
elif menu == "Deteksi Sampah":
    st.title("📷 Deteksi Jenis Sampah")

    uploaded = st.file_uploader("Unggah gambar sampah", type=["jpg", "jpeg", "png"])
    camera = st.camera_input("Atau ambil gambar langsung")

    image = None
    if uploaded:
        image = Image.open(uploaded)
    elif camera:
        image = Image.open(camera)

    if image:
        st.image(image, caption="Gambar yang Anda unggah", use_column_width=True)
        if st.button("Deteksi Sampah"):
            label, confidence = predict(image)
            st.success(f"🔍 Jenis sampah terdeteksi: **{label.upper()}** ({confidence*100:.2f}%)")

            info = waste_info.get(label.lower(), {})
            st.markdown(f"**♻️ Kategori**: {info.get('kategori', '-')}")
            st.markdown(f"**🔁 Daur Ulang**: {info.get('daur_ulang', '-')}")
            st.markdown(f"**📘 Edukasi**: {info.get('edukasi', '-')}")
        else:
            st.info("Klik tombol 'Deteksi Sampah' untuk memulai.")

# Edukasi
elif menu == "Jenis Sampah":
    st.title("📚 Jenis Sampah & Edukasi")
    for jenis, detail in waste_info.items():
        st.subheader(jenis.capitalize())
        st.markdown(f"**Kategori**: {detail['kategori']}")
        st.markdown(f"**Daur Ulang**: {detail['daur_ulang']}")
        st.markdown(f"**Tips/Edukasi**: {detail['edukasi']}")
        st.markdown("---")

# Tentang
else:
    st.title("👥 Tentang Kami")
    st.markdown("""
    Kami adalah kelompok mahasiswa yang peduli terhadap lingkungan.  
    Tujuan kami membuat aplikasi ini adalah untuk meningkatkan kesadaran masyarakat akan pentingnya pengelolaan sampah.

    **Anggota Kelompok:**
    - 🌱 Nama 1
    - 🌱 Nama 2
    - 🌱 Nama 3

    Terima kasih telah menggunakan aplikasi ini!
    """)

