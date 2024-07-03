from flask import Flask, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)

# Initialize variables for embeddings and files
vector_store = None
text_chunks = None
chain = None
initial_instructions = (
        "You are a chatbot in a medical app made by a developers team called CareCode from Suez University"
        "your name is CareChat and Your role is to provide helpful and accurate information on mental health, symptoms, treatments, and general medical advice. "
        "Please be polite, empathetic, and professional in your responses."
    )
# Function to load documents and embeddings (called once on startup)
def load_documents_and_embeddings():
    global vector_store, text_chunks, chain
    print("Loading starts")
    loader = DirectoryLoader('F:/My_Projects/medical_Chatbot/books1/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)

    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
    texts = [str(chunk) for chunk in text_chunks]

    if texts:
        embeddings_list = embeddings.embed_documents(texts)
        print(f"Number of embeddings generated: {len(embeddings_list)}")
    else:
        print("No text chunks to process.")

    vector_store = FAISS.from_documents(text_chunks, embeddings)
    print("Vector Space DONE")

    # Initialize LLM model
    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama",
                        config={'max_new_tokens': 128, 'temperature': 0.01})
    print("Model initialized!!!!!!")

    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Define initial instructions for the model
    

    # Initialize conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )

# Load documents and embeddings on application startup
load_documents_and_embeddings()

# Endpoint for chatbot API
@app.route('/chat', methods=['POST'])
def chat():
    global chain
    
    data = request.json
    query = data['query']
    print(query)
    # Combine initial instructions and the user query
    combined_query = initial_instructions + "\n" + query

    # Truncate the combined query if it exceeds the model's context length
    max_tokens = 512
    combined_tokens = combined_query.split()
    if len(combined_tokens) > max_tokens:
        combined_query = ' '.join(combined_tokens[:max_tokens])

    result = chain({"query": combined_query, "chat_history": []})
    response = {
        "query": query,
        "answer": result["answer"]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)