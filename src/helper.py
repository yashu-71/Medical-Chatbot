import os
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_splits(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def create_faiss_index(text_chunks, embeddings_model):
    texts = [t.page_content for t in text_chunks]
    embeddings = [embeddings_model.encode(text) for text in texts]
    embeddings_array = np.array(embeddings).astype('float32')
    d = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_array)
    return index, texts

def save_embeddings_and_index(embeddings_array, index, embeddings_path, index_path):
    np.save(embeddings_path, embeddings_array)
    faiss.write_index(index, index_path)
    print(f"Embeddings saved to {embeddings_path}")
    print(f"FAISS index saved to {index_path}")

def load_embeddings_and_index(embeddings_path, index_path):
    embeddings_array = np.load(embeddings_path)
    index = faiss.read_index(index_path)
    print(f"Embeddings loaded from {embeddings_path}")
    print(f"FAISS index loaded from {index_path}")
    return embeddings_array, index

def ensure_embeddings_and_index(data_path, embeddings_path, index_path):
    if os.path.exists(embeddings_path) and os.path.exists(index_path):
        return load_embeddings_and_index(embeddings_path, index_path)
    else:
        extracted_data = load_pdf(data_path)
        text_chunks = text_splits(extracted_data)
        embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        index, texts = create_faiss_index(text_chunks, embeddings_model)
        embeddings_array = np.array([embeddings_model.encode(text) for text in texts]).astype('float32')
        save_embeddings_and_index(embeddings_array, index, embeddings_path, index_path)
    return embeddings_array, index

def search(query, embeddings_model, index, texts, k=5):
    query_embedding = embeddings_model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding, k)
    results = [(texts[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results
