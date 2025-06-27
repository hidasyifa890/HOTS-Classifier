
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Konfigurasi halaman
st.set_page_config(
    page_title='Klasifikasi Soal Taksonomi Bloom',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon="📘"
)

# Load model dan vectorizer
model = joblib.load("best_model_svm.pkl")
vectorizer = joblib.load("best_vectorizer_svm.pkl")

# Fungsi klasifikasi
def classify_text(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return prediction

# Navigasi menu dengan tombol
st.title("📘 Aplikasi Klasifikasi Soal Taksonomi Bloom")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("📋 Beranda"):
        st.session_state.page = "beranda"
with col2:
    if st.button("📝 Panduan Penggunaan"):
        st.session_state.page = "panduan"
with col3:
    if st.button("ℹ️ Tentang"):
        st.session_state.page = "tentang"

# Inisialisasi halaman default
if 'page' not in st.session_state:
    st.session_state.page = "beranda"

# Halaman Beranda
if st.session_state.page == "beranda":
    st.header("Klasifikasikan teks Anda di sini")
    input_text = st.text_area("Masukkan soal atau pertanyaan:", height=150)

    if st.button("🔍 Predict"):
        if input_text:
            result = classify_text(input_text)
            st.success(f"Hasil klasifikasi: **{result.upper()}**")
        else:
            st.warning("Silakan masukkan teks terlebih dahulu.")

    # Upload CSV
    st.markdown("---")
    st.subheader("📂 Atau upload file soal (.csv) maksimal 100 baris")
    uploaded_file = st.file_uploader("Upload file CSV dengan kolom 'Pertanyaan'", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if 'Pertanyaan' not in df.columns:
            st.error("❌ Kolom 'Pertanyaan' tidak ditemukan dalam file.")
        else:
            preview = df[['Pertanyaan']].head(100)
            st.write("📋 Preview soal (maksimal 100):")
            st.dataframe(preview)

            # Klasifikasi seluruh data
            vectors = vectorizer.transform(preview['Pertanyaan'].astype(str))
            predictions = model.predict(vectors)
            preview['Hasil Klasifikasi'] = predictions

            st.success("✅ Klasifikasi selesai!")
            st.dataframe(preview)

            # Unduh hasil
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_download = convert_df(preview)
            st.download_button(
                label="📥 Download Hasil",
                data=csv_download,
                file_name='hasil_klasifikasi.csv',
                mime='text/csv'
            )

# Halaman Panduan
elif st.session_state.page == "panduan":
    st.header("📝 Panduan Penggunaan")
    st.markdown("""
    1. Masukkan soal secara manual di kolom teks, lalu klik **Predict** untuk mendapatkan hasil klasifikasi.
    2. Atau unggah file CSV dengan kolom bernama **Pertanyaan**.
    3. Sistem akan mengklasifikasikan setiap soal menjadi kategori **HOTS** atau **LOTS**.
    4. Hasil klasifikasi dapat diunduh dalam bentuk file CSV.
    """)

# Halaman Tentang
elif st.session_state.page == "tentang":
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dikembangkan sebagai bagian dari penelitian klasifikasi soal berdasarkan level kognitif Taksonomi Bloom.
    Algoritma yang digunakan adalah **Support Vector Machine (SVM)**.
    Dibuat menggunakan Python dan Streamlit.
    """)

