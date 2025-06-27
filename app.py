import pickle
import streamlit as st
import random
from PIL import Image


@st.cache_data()
def load_pickled_objects():
    pickled_vector = pickle.load(
        open('temp/model/bestModelKNNSTDEVS0-accTesting87%.pkl', 'rb'))
    pickled_model = pickle.load(
        open('temp/model/bestVectorKNNSTDEV.S0-accTesting87%.pkl', 'rb'))
    return pickled_vector, pickled_model


def main():
    session_state = st.session_state
    if 'text_inputs' not in session_state:
        session_state.text_inputs = [""]  # Default: 1 input box
    if 'predict_button_clicked' not in session_state:
        session_state.predict_button_clicked = False

    image_icon = Image.open('temp/icon/icon2.png')
    st.set_page_config(
        page_title='HOTS Classifier',
        layout='wide',
        initial_sidebar_state='auto',
        page_icon=image_icon
    )

    st.sidebar.title('Klasifikasi Soal HOTS dan LOTS')
    st.sidebar.image(image_icon)
    choice = st.sidebar.selectbox(
        'Main Menu', ['Beranda', 'Panduan Penggunaan', 'Tentang'])
    st.sidebar.info(
        'Web ini dapat melakukan fungsi klasifikasi teks ke dalam kategori HOTS dan LOTS.')

    if choice == 'Panduan Penggunaan':
        st.markdown('<h1>Panduan Penggunaan</h1>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("1. Masuk ke halaman beranda.")
        st.markdown(
            "2. Masukan soal dalam bentuk teks berbahasa Indonesia atau bisa multiple input dalam format csv.")
        st.markdown(
            "3. Gunakan tombol browse file untuk menginput data dalam bentuk csv lalu pilih file csv yang berisi pertanyaan")
        st.markdown("4. Klik tombol predict.")
        st.markdown("5. Akan tampil hasil klasifikasi soal untuk semua input.")

    if choice == 'Beranda':
        
        st.markdown("---")
        st.header("Klasifikasikan teks anda disini!")

        # Container untuk input dinamis
        input_container = st.container()

        # Tombol tambah/hapus input
      #  col1, col2 = st.columns([1, 10])
       
#       with col1:
 #           if st.button('➕ Tambah Input'):
  #              session_state.text_inputs.append("")
   #     with col2:
    #        if st.button('➖ Hapus Input') and len(session_state.text_inputs) > 1:
     #           session_state.text_inputs.pop()

        # Render semua text input
        for i, text in enumerate(session_state.text_inputs):
            session_state.text_inputs[i] = input_container.text_area(
                f" ",
                value=text,
                placeholder=f"Masukkan teks soal",
                key=f"text_input_{i}"
            )

        predict_button = st.button('Predict', key='predict', type='primary')

        if predict_button or session_state.predict_button_clicked:
            session_state.predict_button_clicked = False
            all_filled = all(text.strip()
                             for text in session_state.text_inputs)

            if all_filled:
                pickled_vector, pickled_model = load_pickled_objects()

                for i, text in enumerate(session_state.text_inputs):
                    with st.expander(f"Hasil Klasifikasi Soal {i+1}", expanded=True):
                        predict_text(text, pickled_vector, pickled_model)
            else:
                st.warning(
                    "Semua input teks harus diisi untuk melakukan klasifikasi!")
        st.markdown("---")
        st.header("Upload File CSV")

        uploaded_file = st.file_uploader("Unggah file .csv berisi kolom 'Pertanyaan' maksimal 100 soal", type=["csv"])
        if uploaded_file is not None:
            import pandas as pd
            try:
                df = pd.read_csv(uploaded_file)
                if 'Pertanyaan' not in df.columns:
                    st.error("⚠️ Kolom 'Pertanyaan' tidak ditemukan. Pastikan nama kolom persis 'Pertanyaan'.")
                else:
                    df = df.head(100)  # Batasi hanya 100 soal
                    st.success(f"{len(df)} soal berhasil dimuat. Menampilkan preview:")
                    st.dataframe(df)

                    pickled_vector, pickled_model = load_pickled_objects()
                    pertanyaan = df['Pertanyaan'].astype(str).str.lower()
                    hasil_vector = pickled_model.transform(pertanyaan)
                    prediksi = pickled_vector.predict(hasil_vector)

                    df['Prediksi'] = prediksi
                    st.subheader("✅ Hasil Klasifikasi:")
                    st.dataframe(df[['Pertanyaan', 'Prediksi']])

                    # Unduh hasil
                    hasil_csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Unduh Hasil Klasifikasi", hasil_csv, "hasil_klasifikasi.csv", "text/csv")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca file: {e}")

    elif choice == 'Tentang':
        st.title('Tentang Aplikasi')
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
            ### 🧠 Algoritma Support Vector Machine
            - Menggunakan Support Vector Machine
            - Akurasi mencapai 93%
            - Dapat menangani klasifikasi dua kelas dengan margin optimal
            """)

        with col_method2:
            st.markdown("""
            ### 📊 TF-IDF Vectorizer
            - Preprocessing teks otomatis
            - Stopword removal bahasa Indonesia
            - N-gram (1-2 kata)
            """)

        # Tim Pengembang
        st.markdown("---")
        st.header("👥 Tim Pengembang")

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
            - ✅ Input bisa 100 soal
            - ✅ Akurasi tinggi (93%)
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
