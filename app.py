import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Dr. Bot: Your AI Medical Assistant")  
st.caption("Dr. Bot is an AI-powered medical assistant that helps users assess their symptoms, gather relevant health details, and provide insightful guidance.")  

llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
vectorstore = Chroma(
    persist_directory="D:\\CODIN PLAYGROUND\\ML-AI\\Krish Naik\\Projects\\Simple Projects\\codefest\\chromaDB", 
    embedding_function=embedding
)
retriever = vectorstore.as_retriever()

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Given a chat history and the latest user question, formulate a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_prompt = (
    "You are Dr. Bot, a compassionate and highly knowledgeable AI medical assistant. Your task is to assess the patient's mental and physical health, "
    "analyze symptoms, and predict possible diseases or underlying causes. "
    "Engage with the user in a professional yet empathetic manner, providing well-researched and accurate medical insights. "
    "Ask relevant follow-up questions to gather crucial health details before offering guidance. "
    "Provide clear, actionable prevention measures and diagnostic insights. "
    "Avoid speculation, misinformation, and hallucinations. "
    "Ensure your responses are ethical, human-like, and aligned with medical best practices.\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Use a default session id for the user session
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = "default-session-id"  # Use a static default session ID

# Retrieve and manage chat history from session state
if f"chat_history_{st.session_state['session_id']}" not in st.session_state:
    st.session_state[f"chat_history_{st.session_state['session_id']}"] = []

chat_history = st.session_state[f"chat_history_{st.session_state['session_id']}"]

user_input = st.text_area("How you are feeling right now:")
if user_input:
    response = rag_chain.invoke({"input": user_input, "chat_history": chat_history, "context": ""})
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "ai", "content": response['answer']})  
    
    
    st.session_state[f"chat_history_{st.session_state['session_id']}"] = chat_history
    
    st.write("Dr. Bot:", response['answer'])
