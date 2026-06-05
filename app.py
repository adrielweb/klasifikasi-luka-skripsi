import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

st.set_page_config(page_title="Asisten Cerdas P3K", page_icon="🩹", layout="centered")

# CSS Minimalis & Palet Biru/Navy
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
        background-color: #f4f7f6;
        color: #1a2a40;
    }
    
    h1 {
        color: #112240;
        text-align: center;
        font-weight: 800;
    }
    
    .subtitle {
        text-align: center;
        color: #495670;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-style: italic;
    }

    .box-saran {
        background-color: #ffffff;
        color: #1a2a40 !important; /* <--- TAMBAHKAN BARIS INI */
        border-left: 5px solid #2a6fdb;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🩹 Asisten P3K Cerdas</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Unggah foto cederamu, biar AI bantu kenali dan kasih saran perawatannya~ ☁️✨</div>", unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    return load_model('Model_FineTuned_Luka.keras')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")

kelas_luka = ['Bengkak', 'Bukan Cedera', 'Lecet', 'Luka Bakar', 'Luka Sayatan', 'Memar']

# Kamus Saran P3K (Berdasarkan Standar Medis Umum & Palang Merah)
saran_p3k = {
    'Bengkak': (
        "🧊 <b>Bengkak (Swelling): Terapkan Metode R.I.C.E</b><br>"
        "• <b>R</b>est: Istirahatkan area yang cedera.<br>"
        "• <b>I</b>ce: Kompres dingin (es dibalut handuk) selama 15-20 menit untuk mengurangi radang.<br>"
        "• <b>C</b>ompression: Balut dengan perban elastis (jangan terlalu ketat).<br>"
        "• <b>E</b>levation: Posisikan area yang bengkak lebih tinggi dari jantung untuk mengurangi penumpukan cairan."
    ),
    'Bukan Cedera': (
        "✨ <b>Kondisi Aman (Normal Tissue):</b><br>"
        "Tidak terdeteksi adanya cedera luar pada epidermis yang memerlukan intervensi P3K. Tetap jaga kebersihan dan hidrasi kulit. Dan Kirimkan foto yang terdapat luka pada anda"
    ),
    'Lecet': (
        "💧 <b>Lecet (Abrasion):</b><br>"
        "• Cuci tangan terlebih dahulu.<br>"
        "• Bersihkan kotoran dari luka perlahan menggunakan air mengalir atau cairan saline (NaCl 0.9%).<br>"
        "• <i>Peringatan Medis:</i> Hindari menuangkan alkohol atau yodium pekat langsung ke luka karena dapat merusak jaringan sehat (sitotoksik) dan memperlambat penyembuhan.<br>"
        "• Keringkan perlahan, oleskan salep antibiotik (jika ada), dan tutup dengan kasa steril."
    ),
    'Luka Bakar': (
        "🚰 <b>Luka Bakar Ringan (Thermal Burn 1st/2nd Degree):</b><br>"
        "• Segera hentikan proses kerusakan jaringan dengan mengaliri luka di bawah air keran biasa (suhu ruang) selama 15-20 menit.<br>"
        "• <i>Peringatan Medis:</i> <b>JANGAN</b> mengoleskan pasta gigi, mentega, kecap, atau es batu! Bahan-bahan tersebut mengunci panas di dalam kulit dan memicu infeksi parah.<br>"
        "• Tutup perlahan dengan kasa steril yang tidak menempel (non-adherent dressing)."
    ),
    'Luka Sayatan': (
        "🩸 <b>Luka Sayatan (Laceration/Incision):</b><br>"
        "• Prioritas utama: Hentikan pendarahan dengan memberikan tekanan langsung (<i>direct pressure</i>) menggunakan kasa steril atau kain bersih selama beberapa menit.<br>"
        "• Setelah darah berhenti, bilas dengan air mengalir untuk mencegah infeksi.<br>"
        "• Jika sayatan terlihat sangat dalam, menganga, atau pendarahan tidak berhenti setelah 10 menit ditekan, segera ke IGD untuk mendapatkan jahitan medis."
    ),
    'Memar': (
        "❄️ <b>Memar (Contusion):</b><br>"
        "• Terjadi karena pecahnya pembuluh darah kapiler di bawah kulit. Berikan kompres dingin pada 24-48 jam pertama untuk menyempitkan pembuluh darah.<br>"
        "• <i>Peringatan Medis:</i> <b>Hindari memijat atau mengurut</b> area yang memar karena dapat merusak pembuluh darah lebih lanjut dan memperparah penyebaran darah di bawah kulit.<br>"
        "• Setelah 48 jam, ganti dengan kompres hangat untuk memperlancar penyerapan sisa darah."
    )
}

uploaded_file = st.file_uploader("Pilih foto cedera (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption='Foto yang diunggah', use_container_width=True)

    with col2:
        with st.spinner('Menganalisis pola visual... 🔍'):
            # Preprocessing
            img_resized = img.resize((224, 224)) 
            img_array = np.array(img_resized)
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Prediksi AI
            prediction = model.predict(img_array)[0]
            
            # Mengurutkan hasil dari yang persentasenya paling besar
            urutan_indeks = np.argsort(prediction)[::-1]
            
            # Mengambil prediksi peringkat 1 (Diagnosis Utama)
            idx_utama = urutan_indeks[0]
            label_utama = kelas_luka[idx_utama]
            akurasi_utama = prediction[idx_utama] * 100

            st.success("Analisis Selesai!")
            st.markdown(f"### Diagnosis Utama:")
            st.markdown(f"#### **{label_utama}** ({akurasi_utama:.1f}%)")
            st.progress(int(akurasi_utama))

            # Menyaring kemungkinan kedua & ketiga (Threshold 25%)
            kemungkinan_lain = []
            for i in range(1, len(urutan_indeks)):
                idx = urutan_indeks[i]
                akurasi = prediction[idx] * 100
                if akurasi >= 25.0: # AMBANG BATAS 25%
                    kemungkinan_lain.append((kelas_luka[idx], akurasi))

            if kemungkinan_lain:
                st.warning("⚠️ Mengindikasi Kondisi Gabungan")
                for label, akurasi in kemungkinan_lain:
                    st.markdown(f"- Kemungkinan sekunder: **{label}** ({akurasi:.1f}%)")
                    
    # Menampilkan Kotak Saran
    st.markdown("---")
    st.markdown("### 🩺 Panduan Penanganan:")
    
    # Menampilkan saran untuk diagnosis utama
    st.markdown(f"<div class='box-saran'>{saran_p3k[label_utama]}</div>", unsafe_allow_html=True)
    
    # Menampilkan saran tambahan jika melewati threshold 25%
    if kemungkinan_lain:
        st.markdown("**Saran Tambahan (Berdasarkan indikasi sekunder):**")
        for label, _ in kemungkinan_lain:
            st.markdown(f"<div class='box-saran'>{saran_p3k[label]}</div>", unsafe_allow_html=True)

    st.warning("🚨 **CATATAN MEDIS PENTING:**\nAplikasi AI ini hanya berfungsi sebagai panduan Pertolongan Pertama pada Kecelakaan (P3K). **Segera kunjungi dokter atau fasilitas kesehatan terdekat** jika luka sangat dalam, pendarahan tidak berhenti, atau kondisi memburuk!")
