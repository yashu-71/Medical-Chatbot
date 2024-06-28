import os
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import faiss
import numpy as np 
from sentence_transformers import SentenceTransformer

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

embeddings = download_hugging_face_embeddings()
embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

texts = [t for t in text_chunks]
embeddings = [embeddings_model.encode(text) for text in texts]

# Convert embeddings to numpy array
embeddings_array = np.array(embeddings).astype('float32')

# Create a FAISS index
d = embeddings_array.shape[1]  # dimension of embeddings
index = faiss.IndexFlatL2(d)  # L2 distance index
index.add(embeddings_array)  # add vectors to the index

print("Data successfully indexed!")
