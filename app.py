
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import base64

# Set ikon dan konfigurasi halaman
image_icon = "📘"
st.set_page_config(
    page_title='HOTS Classifier',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon=image_icon
)

st.title("📘 Aplikasi Klasifikasi Soal Taksonomi Bloom")

# Load model dan vectorizer
model = joblib.load("model_svm.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Fungsi klasifikasi
def classify_text(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return prediction

# Halaman utama
st.header("Klasifikasikan teks anda disini!")

input_text = st.text_area("Masukkan soal atau pertanyaan:", height=150)

if st.button("🔍 Predict"):
    if input_text:
        result = classify_text(input_text)
        st.success(f"Hasil klasifikasi: **{result.upper()}**")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")

# Upload CSV setelah prediksi
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
