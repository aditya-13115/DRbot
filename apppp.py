import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import chromadb
import os

from dotenv import load_dotenv
load_dotenv()

# Ensure ChromaDB directory exists
CHROMA_DB_PATH = "./chroma_db"
if not os.path.exists(CHROMA_DB_PATH):
    os.makedirs(CHROMA_DB_PATH)

# Load environment variables
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
api_key = os.getenv("GROQ_API_KEY")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Dr. Bot: Your AI Medical Assistant")  
st.caption("Upload PDFs and get them summarized.")  

if api_key:
    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

    # Initialize ChromaDB client and collection
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    vectorstore = Chroma(client=chroma_client, collection_name="medical_data", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    # Session management
    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = "./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Add documents to the existing ChromaDB collection
        vectorstore.add_documents(splits)
        
        # Update retriever after adding new docs
        retriever = vectorstore.as_retriever()

    # Prompt templates
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, rephrase it as a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are a medical chatbot named Dr. Bot. "
        "Your task is to summarize the provided medical documents and answer health-related questions. "
        "Provide factual, well-researched, and human-like responses based on the retrieved context. "
        "Do not hallucinate or speculate. If you do not know an answer, say so. "
        "You are a friendly and professional medical assistant."
        "\n\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # User input for chat
    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        st.write("Dr. Bot:", response['answer'])

        # Expandable chat history and retrieved context
        with st.expander("View Chat History and Context"):
            st.write("Chat History:", session_history.messages)
            st.write("Retrieved Context:", response.get("context", "No context available"))

else:
    st.warning("Please enter a valid API key in your environment variables.")
