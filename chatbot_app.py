import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import gdown
import os

# URL Google Drive untuk file chatbot_data.pkl
GDRIVE_URL = "https://drive.google.com/uc?id=1PbTbPboHnqs-eCr63gzYrTC1Ub0XwSaw"

# Fungsi untuk mendownload file dari Google Drive
def download_from_gdrive(file_url, output_path):
    try:
        gdown.download(file_url, output_path, quiet=False)
        return True
    except Exception as e:
        st.error(f"Gagal mendownload file dari Google Drive: {e}")
        return False

# Fungsi untuk memuat data
def load_data(filepath='chatbot_data.pkl'):
    with open(filepath, 'rb') as f:
        index, sentence_model, sentences, summaries = pickle.load(f)
    return index, sentence_model, sentences, summaries

# Fungsi chatbot
def chatbot(queries, index, sentence_model, sentences, summaries, top_k=3):
    query_embeddings = sentence_model.encode(queries)
    D, I = index.search(query_embeddings, k=top_k)

    responses = []
    for query, indices in zip(queries, I):
        relevant_sentences = []
        relevant_summaries = []

        for idx in indices:
            if 0 <= idx < len(sentences):
                relevant_sentences.append(sentences[idx])
                summary = summaries[idx] if idx < len(summaries) and summaries[idx] != "Ringkasan tidak tersedia." else ""
                if summary:
                    relevant_summaries.append(summary)

        if relevant_sentences:
            combined_sentences = " ".join(relevant_sentences[:2])
            combined_summaries = " ".join(relevant_summaries[:2]) if relevant_summaries else "Tidak ada ringkasan yang relevan."

            response = (
                f"**Pertanyaan:** {query}\n\n"
                f"**Jawaban:** {combined_sentences}\n\n"
                f"{f'**Ringkasan:** {combined_summaries}' if relevant_summaries else ''}"
            )
        else:
            response = f"**Pertanyaan:** {query}\n\n**Jawaban:** Tidak ada konten relevan yang ditemukan."

        responses.append(response)
    return responses

# Streamlit App
def main():
    st.title("Chatbot AI")
    st.write("Interaksi dengan dokumen Anda menggunakan AI.")

    # File lokal
    local_path = "chatbot_data.pkl"

    # Cek apakah file lokal ada, jika tidak, download dari Google Drive
    if not os.path.exists(local_path):
        st.info("File data tidak ditemukan. Mendownload dari Google Drive...")
        if download_from_gdrive(GDRIVE_URL, local_path):
            st.success("File berhasil didownload.")
        else:
            st.error("Gagal mendownload file. Periksa koneksi internet Anda.")
            st.stop()

    # Memuat data chatbot
    try:
        index, sentence_model, sentences, summaries = load_data(local_path)
        st.sidebar.success("Data berhasil dimuat.")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat data: {e}")
        st.stop()

    # Input pertanyaan pengguna
    queries = st.text_area("Masukkan pertanyaan Anda (pisahkan dengan ';' untuk pertanyaan ganda):")
    if st.button("Ajukan Pertanyaan"):
        if not queries.strip():
            st.warning("Masukkan setidaknya satu pertanyaan.")
        else:
            queries_list = queries.split(";")
            responses = chatbot(queries_list, index, sentence_model, sentences, summaries)
            for response in responses:
                st.markdown(response)

if __name__ == '__main__':
    main()
