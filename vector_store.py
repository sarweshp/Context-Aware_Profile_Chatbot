from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

class VectorStoreManager:
    def __init__(self, model="models/embedding-001"):
        """
        Initialize the Vector Store Manager.
        
        Args:
            model (str, optional): Embedding model to use. Defaults to "models/embedding-001".
        """
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=1000,
            separators=["\n\n", "\n", " ", ""]
        )

    def create_vector_store(self, text, index_path=None):
        """
        Create a FAISS vector store from given text.
        
        Args:
            text (str): Text to be embedded and stored.
            index_path (str, optional): Path to save the FAISS index.
        
        Returns:
            FAISS: Created vector store.
        """
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(text)
        
        # Create vector store
        vector_store = FAISS.from_texts(
            texts=text_chunks,
            embedding=self.embeddings
        )
        
        # Save index if path is provided
        if index_path:
            vector_store.save_local(index_path)
        
        return vector_store

    def retrieve_relevant_chunks(self, query, vector_store, top_k=5):
        """
        Retrieve top_k relevant chunks from the vector store.
        
        Args:
            query (str): User's question.
            vector_store (FAISS): The FAISS vector store.
            top_k (int, optional): Number of top similar chunks. Defaults to 5.
        
        Returns:
            str: Retrieved context.
        """
        # Retrieve similar documents
        similar_docs = vector_store.similarity_search(query, k=top_k)
        
        # Extract the text content from the retrieved documents
        retrieved_chunks = [doc.page_content for doc in similar_docs]
        context = "\n\n".join(retrieved_chunks)
        
        return context