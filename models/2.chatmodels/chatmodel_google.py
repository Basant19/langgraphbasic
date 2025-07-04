import streamlit as st
from dotenv import load_dotenv
import os

# LangChain v0.3+ compatible
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings.base import Embeddings

# ✅ Fully local sentence-transformer
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Gemini Document Chatbot")
st.header("📄 Gemini Document Search Chatbot 🔍")

# Load document
loader = TextLoader(
    r"D:\Generative_ai_practise\Basic_model_setup\projects\sample1.txt",
    encoding="utf-8"
)
docs = loader.load()

# Show stats
num_total_characters = sum(len(doc.page_content) for doc in docs)
st.write(f"Loaded {len(docs)} documents. Avg characters: {num_total_characters / len(docs):,.0f}")

# ✅ Custom wrapper class around SentenceTransformer to act as LangChain Embedding
class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Use custom wrapper
embeddings = LocalSentenceTransformerEmbeddings()

# FAISS vector store
docsearch = FAISS.from_documents(docs, embeddings)

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=google_api_key
)

# RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=docsearch.as_retriever(),
    chain_type="stuff"
)

# User input
query = st.text_input("Ask a question:")

if st.button("Search"):
    if query.strip():
        response = qa_chain.run(query)
        st.subheader("Answer:")
        st.write(response)
    else:
        st.warning("Please enter a question.")
