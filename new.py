import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    loader = TextLoader(uploaded_file)
    documents = str(loader.load())
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    retriever = db.as_retriever()
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
    return qa.run(query_text)

uploaded_file = '/Users/benstager/Desktop/11.txt'
openai_api_key = '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'
query_text = 'What is the main idea?'

print(generate_response(uploaded_file, openai_api_key, query_text))