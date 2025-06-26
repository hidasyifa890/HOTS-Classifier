import streamlit as st
import pandas as pd
import pickle

# Header Aplikasi
st.markdown("<h1 style='text-align: center; color: navy;'>📚 Klasifikasi Soal HOTS & LOTS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Aplikasi ini digunakan untuk mengklasifikasikan soal berdasarkan Taksonomi Bloom menggunakan model SVM.</p>", unsafe_allow_html=True)
st.markdown("---")

# Informasi Model
st.sidebar.header("📊 Informasi Model")
st.sidebar.markdown("- Model: Support Vector Machine (SVM)")
st.sidebar.markdown("- Akurasi: 93.33%")
st.sidebar.markdown("- F1-Score: 93% (macro average)")
st.sidebar.markdown("---")
st.sidebar.markdown("🧠 **Kategori:**")
st.sidebar.markdown("- HOTS = Higher Order Thinking Skills")
st.sidebar.markdown("- LOTS = Lower Order Thinking Skills")
st.sidebar.markdown("---")
('Tentang Aplikasi')
        st.markdown("---")

        # Deskripsi Aplikasi
        st.header("📚 Tentang K-Bloom Classifier")
        st.markdown("""
        <div style='text-align: justify;'>
        K-Bloom Classifier adalah aplikasi berbasis web yang dirancang untuk membantu pendidik dalam menganalisis 
        dan mengklasifikasikan soal berdasarkan level kognitif Taksonomi Bloom. Aplikasi ini dapat membedakan soal 
        HOTS (Higher Order Thinking Skills) dan LOTS (Lower Order Thinking Skills) secara otomatis menggunakan 
        teknik machine learning.
        </div>
        """, unsafe_allow_html=True)

        # Metode dalam 2 kolom
        st.markdown("---")
        st.header("🔬 Metode yang Digunakan")

        col_method1, col_method2 = st.columns(2)

        with col_method1:
            st.markdown("""
            ### 🧠 Algoritma Decision Tree
            - Menggunakan Decision Tree 
            - Akurasi mencapai 93%
            - Optimasi parameter menggunakan grid search
            """)

        with col_method2:
            st.markdown("""
            ### 📊 TF-IDF Vectorizer
            - Preprocessing teks otomatis
            - Stopword removal bahasa Indonesia
            - N-gram (1-3 kata)
            """)

        # Tim Pengembang
        st.markdown("---")
        st.header("👥 Tim Pengembang Inti")

        dev_col1, dev_col2 = st.columns([1, 3])


        with dev_col2:
            st.markdown("""
            <div style='background-color:#f0f2f6; padding:20px; border-radius:10px;'>
                <h3 style='color:#2e86c1;'>Pengembang Utama</h3>
                <p><b>Nama:</b> Hida Syifaurohmah</p>
                <p><b>Pembimbing:</b> Dr. Ir. Fatchul Arifin, M.T.</p>
                <p><b>Institusi:</b> Program Pascasarjana Universitas Negeri Yogyakarta</p>
                <p><b>Tahun:</b> 2025</p>
            </div>
            """, unsafe_allow_html=True)

        # Fitur Aplikasi
        st.markdown("---")
        st.header("✨ Fitur Unggulan")

        feature_col1, feature_col2 = st.columns(2)

        with feature_col1:
            st.markdown("""
            - ✅ Multi-input teks sekaligus
            - ✅ Antarmuka interaktif
            - ✅ Rekomendasi peningkatan soal
            """)

        with feature_col2:
            st.markdown("""
            - ✅ Hasil real-time
            - ✅ Input dinamis (tambah/hapus)
            - ✅ Akurasi tinggi (87%)
            """)

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center;color:#5d6d7e;'>© 2023 K-Bloom Classifier | Universitas Negeri Yogyakarta</p>", unsafe_allow_html=True)


def predict_text(text, vectorizer, model):
    lots_suggest = [
        'Sebaiknya soal mengandung kemampuan atau keterampilan membedakan, mengorganisasikan, dan menghubungkan. Kata kerja operasional yang biasa digunakan adalah membandingkan, mengkritisi, mengurutkan, membedakan, dan menentukan.',
        'Sebaiknya soal mengandung kata kerja operasional yang digunakan yaitu mengevaluasi, memilih, menilai, menyanggah, dan memberikan pendapat.',
        'Sebaiknya soal mengandung kemampuan dalam merancang, membangun, merencanakan, memproduksi,  menemukan, dan menyempurnakan. Kata kerja operasional yang digunakan adalah memperjelas, menafsirkan, dan memprediksi.'
    ]

    sentence = [text]
    vectorized_text = model.transform(sentence)
    predict = vectorizer.predict(vectorized_text)

    predicted_class = predict[0]

    if predicted_class == 'Lower Order Thinking Skills':
        random_suggestion = random.choice(lots_suggest)
        st.info(f'Teks diprediksi sebagai {predicted_class}.')
        st.info(f'Saran : {random_suggestion}')
    else:
        st.info(f'Teks diprediksi sebagai {predicted_class}.')


if __name__ == '__main__':
    main()
