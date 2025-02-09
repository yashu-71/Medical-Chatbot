Medical Chatbot 
A health and medical question-answering chatbot designed to provide reliable and accessible healthcare information using Large Language Models (LLMs). 
 

Features 
1.Answers medical and health-related queries 
2.Provides symptom-based suggestions 
3.Retrieves medical knowledge using FAISS for better accuracy  
4.User-friendly chatbot interface  
5.Expandable & customizable for future improvements  

  

Tech Stack 
1. Python 
2. Flask (Backend)  
3. FAISS (Vector search for knowledge retrieval)  
4. Hugging Face Embeddings (Text representation)  
5. LangChain (LLM-powered QA system)  
6. CTransformers (Llama-based LLM model)  
7. HTML/CSS + JavaScript (For Web UI)  

  

Installation and Usage 


1. Clone the repository  
       git clone https://github.com/yashu-71/Medical-Chatbot 

2. Install the required packages  
     pip install -r requirements.txt 

3. Run the application  
     python app.py 

4. Access the chatbot interface  
   http://localhost:5000 

  

Data Processing and Retrieval 

1. PDF Extraction: Loads medical knowledge from PDFs using helper functions. 
2. Text Splitting: Splits extracted text into smaller chunks. 
3. Vector Embeddings: Converts text into numerical embeddings using Hugging Face models. 
4. FAISS Indexing: Stores and retrieves information efficiently. 
5. LLM Response Generation: Uses Llama-2 (via CTransformers) to generate accurate answers. 

  

Future Enhancements 
1.Voice-based interactions 
2.More efficient search and retieval using advanced FAISS tuning 

  

Contributing 
Fell free to fork this repository and contribute!Pull requests are welcome. 

 
