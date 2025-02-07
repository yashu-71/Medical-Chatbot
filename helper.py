
import os
import faiss
import numpy as np
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pdf(data):
    logging.info("Loading PDFs from directory: %s", data)
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    logging.info("Loaded %d documents", len(documents))
    return documents

def text_splits(extracted_data):
    logging.info("Splitting documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    logging.info("Created %d text chunks", len(text_chunks))
    return text_chunks

def create_and_save_faiss_index(text_chunks, index_path="faiss_index/index.faiss", texts_path="faiss_index/texts.pkl"):
    logging.info("Creating and saving FAISS index")
    embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    texts = [t.page_content for t in text_chunks]
    embeddings = [embeddings_model.encode(text) for text in texts]

    embeddings_array = np.array(embeddings).astype('float32')
    d = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_array)

    # Save the FAISS index
    faiss.write_index(index, index_path)
    logging.info("FAISS index saved at %s", index_path)

    # Save texts and embeddings model
    with open(texts_path, 'wb') as f:
        pickle.dump((texts, embeddings_model), f)
    logging.info("Texts and embeddings model saved at %s", texts_path)

    return index, texts, embeddings_model

def load_faiss_index(index_path="faiss_index/index.faiss", texts_path="faiss_index/texts.pkl"):
    logging.info("Loading FAISS index from %s", index_path)
    index = faiss.read_index(index_path)
    logging.info("FAISS index loaded")

    logging.info("Loading texts and embeddings model from %s", texts_path)
    with open(texts_path, 'rb') as f:
        texts, embeddings_model = pickle.load(f)
    logging.info("Texts and embeddings model loaded")

    return index, texts, embeddings_model
