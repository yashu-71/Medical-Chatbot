from langchain import PromptTemplate

prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
chain_type_kwargs = {"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
# PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# chain_type_kwargs = {"prompt": PROMPT}
