import streamlit as st
import os
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
llm=ChatOpenAI(model="gpt-4.1-2025-04-14")
prompt=ChatPromptTemplate.from_template(
    """ Answer the questions based on the provided context only.
    Please be accurate in answering the question based on the context provided.
    <context>
    {context}
    <context>
    Question:{input}
"""
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        loader=PyPDFDirectoryLoader("data/research_papers")
        docs=loader.load()
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        split_docs=text_splitter.split_documents(docs)
        embeddings=OpenAIEmbeddings()
        st.session_state.vectors=FAISS.from_documents(split_docs,embeddings)

user_prompt=st.text_input("Enter your query about the research papers you provided")

if st.button("Create Digest"):
    create_vector_embeddings()
    st.write("Summary is ready, ask away !")

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({"input":user_prompt})

    st.write(response['answer'])


