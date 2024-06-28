from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from src.prompt import *
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains.retrieval_qa.base import BaseRetriever
from langchain.schema import Document
from pydantic import BaseModel, Field
from typing import Callable, List
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer # type: ignore
import os

def load_pdf(data):
    loader=DirectoryLoader(data,
                           glob="*.pdf",
                           loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents


extracted_data = load_pdf("S:\Projects\Medical-Chat-Bot\Data")
embeddings = download_hugging_face_embeddings()
embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Text Chunks

def text_splits(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)

    return text_chunks

text_chunks=text_splits(extracted_data)
texts = [t.page_content for t in text_chunks]
embeddings = [embeddings_model.encode(text) for text in texts]

# Convert embeddings to numpy array
embeddings_array = np.array(embeddings).astype('float32')

# Create a FAISS index
d = embeddings_array.shape[1]  # dimension of embeddings
index = faiss.IndexFlatL2(d)  # L2 distance index
index.add(embeddings_array)  # add vectors to the index

class MyCustomRetriever(BaseRetriever):
    search_function: Callable[[str, int], List[str]] = Field(..., exclude=True)

    def __init__(self, search_function: Callable[[str, int], List[str]]):
        super().__init__()
        self.search_function = search_function

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Call your search function to get search results
        search_results = self.search_function(query, k=2)
        # Process search results to match the expected format
        processed_data = [Document(page_content=result[0], metadata={"score": result[1]}) for result in search_results]
        return processed_data
    
llm=CTransformers(model="S:\Projects\Medical-Chat-Bot\Model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

# Define your custom search function
def search(query: str, k: int = 2):
    query_embedding = embeddings_model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding, k)
    results = [(texts[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

# Create an instance of your custom retriever
custom_retriever = MyCustomRetriever(search_function=search)

# Create the RetrievalQA instance using your custom retriever
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=custom_retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')



# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     result=qa({"query": input})
#     print("Response : ", result["result"])
#     return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
