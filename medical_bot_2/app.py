import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
import os

# URL to the PDF file on GitHub
pdf_url = "https://raw.githubusercontent.com/Nancy2305/Mental_health_chatbot/main/medical_bot_2/mental_health_Document.pdf"
local_filename = "mental_health_Document.pdf"

# Download the PDF file
response = requests.get(pdf_url)
with open(local_filename, 'wb') as f:
    f.write(response.content)

# Load the PDF using DirectoryLoader
loader = DirectoryLoader('.', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()


if 'history' not in st.session_state:
    st.session_state['history'] = []

# Load the pdf files from the path
loader = DirectoryLoader(r'mental_health_Document.pdf', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split text into chunks
text_splitter  = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device':"cpu"})

# Vectorstore
vector_store = FAISS.from_documents(text_chunks, embeddings)

from huggingface_hub import login
login(token='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')

# Create LLM by loading from Hugging Face directly
#model_name = "meta-llama/Llama-2-7b-chat-hf"  # Hugging Face model repo name
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
# Use a publicly accessible model
model_name = "facebook/opt-6.7b"  # Open and accessible without gating
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)


# Memory to store chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Conversational retrieval chain
chain = ConversationalRetrievalChain.from_llm(llm=model, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory)

st.title("HealthCare ChatBot 🧑🏽‍⚕️")

def conversation_chat(query):
    try:
        result = chain({"question": query, "chat_history": memory.load_memory_variables({})['chat_history']})
        memory.save_context({"input": query}, {"output": result["answer"]})  # Save context in memory
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return "Sorry, I couldn't process your request."

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Mental Health", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state['generated'][i], key=str(i), avatar_style="fun-emoji")

# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()
