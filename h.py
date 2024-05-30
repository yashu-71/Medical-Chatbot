from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from typing import List, Dict

# Function to load PDFs
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Extract data from PDFs
extracted_data = load_pdf("data/")

# Function to split text into chunks
def text_splits(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Split text into chunks
text_chunks = text_splits(extracted_data)
print("Length of my chunk", len(text_chunks))

# Function to download HuggingFace embeddings model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# Get embeddings model
embeddings_model = download_hugging_face_embeddings()
embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Verify embedding model works
query_result = embeddings_model.encode("Hello world", convert_to_tensor=False)
print("Length", len(query_result))

# Convert text chunks to embeddings
texts = [t.page_content for t in text_chunks]
embeddings = embeddings_model.encode(texts, convert_to_tensor=False)

# Convert embeddings to numpy array
embeddings_array = np.array(embeddings).astype('float32')

# Create a FAISS index
d = embeddings_array.shape[1]  # dimension of embeddings
index = faiss.IndexFlatL2(d)  # L2 distance index
index.add(embeddings_array)  # add vectors to the index

print("Data successfully indexed!")

# Function to perform similarity search
def search(query, k=5):
    query_embedding = embeddings_model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding, k)  # search
    results = [(texts[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

# Example search
query = "Example query text."
results = search(query, k=3)
for result in results:
    print(f"Text: {result[0]}, Distance: {result[1]}")

# Initialize your language model
llm = CTransformers(model="gpt-3.5-turbo")

# Define additional chain type kwargs if needed
chain_type_kwargs = {}

# Initialize the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=None,  # No retriever needed
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
)

# Perform a query
qa_query = "What are Allergies?"
qa_result = qa({"query": qa_query})

# Print the results
print("QA Result:", qa_result["result"])
print("Source Documents:", qa_result["source_documents"])
