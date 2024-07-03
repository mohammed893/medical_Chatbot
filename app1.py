import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate

# Function to load documents and embeddings, with caching
@st.cache_resource
def load_documents_and_embeddings():
    loader = DirectoryLoader('F:/My_Projects/medical_Chatbot/books1/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)
    
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
    
    # Ensure text_chunks are strings
    texts = [str(chunk) for chunk in text_chunks]
    
    if texts:
        embeddings_list = embeddings.embed_documents(texts)
        print(f"Number of embeddings generated: {len(embeddings_list)}")
    else:
        print("No text chunks to process.")
    
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    print("Vector Space DONE")
    
    return vector_store, text_chunks

# Load documents and embeddings
vector_store, text_chunks = load_documents_and_embeddings()

# Initialize LLM model
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama",
                    config={'max_new_tokens': 128, 'temperature': 0.01})
print("Model initialized!!!!!!")

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define initial instructions for the model
initial_instructions = (
    "You are a chatbot in a medical app made by a developers team called CareCode from Suez  "
    "your name is CareChat and Your role is to provide helpful and accurate information on mental health, symptoms, treatments, and general medical advice. "
    "Please be polite, empathetic, and professional in your responses."
)

# Initialize conversational retrieval chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    chain_type='stuff',
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    memory=memory
)

# Streamlit app title and session state initialization
st.title("HealthCare ChatBot 🧑🏽‍⚕️")

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

# Function to handle conversation chat
def conversation_chat(query):
    # Combine initial instructions and the user query
    combined_query = initial_instructions + "\n" + query

    # Truncate the combined query if it exceeds the model's context length
    max_tokens = 512
    combined_tokens = combined_query.split()
    if len(combined_tokens) > max_tokens:
        combined_query = ' '.join(combined_tokens[:max_tokens])

    result = chain({"question": combined_query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Function to display chat history
def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Mental Health", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Initialize session state
initialize_session_state()

# Display chat history
display_chat_history()