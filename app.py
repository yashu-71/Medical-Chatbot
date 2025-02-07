from flask import Flask, request, jsonify, render_template
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from pydantic import BaseModel, Field
from typing import Callable, List
from helper import load_pdf, text_splits, create_and_save_faiss_index, load_faiss_index
import os
from langchain import PromptTemplate
import numpy as np
from langchain.chains.retrieval_qa.base import BaseRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

class MyCustomRetriever(BaseRetriever):
    search_function: Callable[[str, int], List[str]] = Field(..., exclude=True)

    def __init__(self, search_function: Callable[[str, int], List[str]]):
        super().__init__()
        self.search_function = search_function

    def _get_relevant_documents(self, query: str) -> List[Document]:
        search_results = self.search_function(query, k=2)
        processed_data = [Document(page_content=result[0], metadata={"score": result[1]}) for result in search_results]
        for doc in processed_data:
            logging.info("Retrieved document: %s", doc.page_content[:200])
        return processed_data

def create_qa_model():
    index_path = r"\project\Medical-Chat-Bot\index.faiss"
    texts_path = r"\project\Medical-Chat-Bot\texts.pkl"

    # Check if the FAISS index and texts exist
    if os.path.exists(index_path) and os.path.exists(texts_path):
        logging.info("Loading existing FAISS index and texts")
        index, texts, embeddings_model = load_faiss_index(index_path, texts_path)
    else:
        logging.info("Creating new FAISS index and texts")
        extracted_data = load_pdf("\project\Medical-Chat-Bot\Data")
        text_chunks = text_splits(extracted_data)
        index, texts, embeddings_model = create_and_save_faiss_index(text_chunks, index_path, texts_path)

    def search(query: str, k: int = 2):
        logging.info("Performing search for query: %s", query)
        query_embedding = embeddings_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        distances, indices = index.search(query_embedding, k)
        results = [(texts[i], distances[0][j]) for j, i in enumerate(indices[0])]
        logging.info("Search completed with %d results", len(results))
        for i, (text, score) in enumerate(results):
            logging.info(f"Result {i + 1}: {text[:200]}... (score: {score})")
        return results

    custom_retriever = MyCustomRetriever(search_function=search)

    llm = CTransformers(
        model="\project\Medical-Chat-Bot\Model\llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config={'max_new_tokens': 512, 'temperature': 0.8}
        # 512 0.8
        # 100,0.9
    )

    prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=custom_retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    logging.info("QA model created successfully")
    return qa

logging.info("Loading model...")
embeddings = HuggingFaceEmbeddings()

try:
    qa_model = create_qa_model()
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to create QA model: {str(e)}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    if not qa_model:
        return jsonify({"error": "QA system is not available"}), 500
    user_input = request.json["query"]
    try:
        result = qa_model.invoke({"query": user_input})
        answer = result.get('result', 'Sorry, I could not find an answer to your question.')
        return jsonify({"answer": answer})
        # return jsonify({"answer": result['result']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
