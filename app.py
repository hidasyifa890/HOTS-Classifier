import pickle
import streamlit as st
import random
from PIL import Image

@st.cache_data()
def load_pickled_objects():
    pickled_vector = pickle.load(open('temp/model/bestModelKNNSTDEVS0-accTesting87%.pkl', 'rb'))
    pickled_model = pickle.load(open('temp/model/bestVectorKNNSTDEV.S0-accTesting87%.pkl', 'rb'))
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
    choice = st.sidebar.selectbox('Main Menu', ['Beranda', 'Panduan Penggunaan','Tentang'])
    st.sidebar.info('Web ini dapat melakukan fungsi klasifikasi teks ke dalam kategori HOTS dan LOTS.')

    if choice == 'Panduan Penggunaan':
        st.markdown('<h1>Panduan Penggunaan</h1>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("1. Masuk ke halaman beranda.")
        st.markdown("2. Masukan soal dalam bentuk teks berbahasa Indonesia (bisa multiple input).")
        st.markdown("3. Gunakan tombol '+' untuk menambah input atau '-' untuk menghapus input.")
        st.markdown("4. Klik tombol predict.")
        st.markdown("5. Akan tampil hasil klasifikasi soal untuk semua input.")

    if choice == 'Beranda':
        st.header("Klasifikasikan teks anda disini!")
        
        # Container untuk input dinamis
        input_container = st.container()
        
        # Tombol tambah/hapus input
        col1, col2 = st.columns([1, 10])
        with col1:
            if st.button('➕ Tambah Input'):
                session_state.text_inputs.append("")
        with col2:
            if st.button('➖ Hapus Input') and len(session_state.text_inputs) > 1:
                session_state.text_inputs.pop()
        
        # Render semua text input
        for i, text in enumerate(session_state.text_inputs):
            session_state.text_inputs[i] = input_container.text_area(
                f"Soal {i+1}",
                value=text,
                placeholder=f"Masukkan teks soal {i+1}",
                key=f"text_input_{i}"
            )

        predict_button = st.button('Predict', key='predict', type='primary')

        if predict_button or session_state.predict_button_clicked:
            session_state.predict_button_clicked = False
            all_filled = all(text.strip() for text in session_state.text_inputs)
            
            if all_filled:
                pickled_vector, pickled_model = load_pickled_objects()
                
                for i, text in enumerate(session_state.text_inputs):
                    with st.expander(f"Hasil Klasifikasi Soal {i+1}", expanded=True):
                        predict_text(text, pickled_vector, pickled_model)
            else:
                st.warning("Semua input teks harus diisi untuk melakukan klasifikasi!")

    elif choice == 'Tentang':
        # ... (kode tentang tetap sama)
        pass

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