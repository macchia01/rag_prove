import os
import psutil
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Faiss invece di Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Hugging Face per embeddings
from langchain_ollama import ChatOllama  # Import corretto per LLM
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class CPUOptimizedRAGPipeline:
    def __init__(self, model_name="llama3.2", embedding_model="sentence-transformers/all-MiniLM-L6-v2", doc_path=None):
        self.setup_logging()
        self.check_memory()
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.doc_path = doc_path

        # Usa Hugging Face embeddings invece di OllamaEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.llm = ChatOllama(model=self.model_name)

        if self.doc_path:
            self.documents = self.load_documents()
            self.vectorstore = self.create_vectorstore()
            self.rag_chain = self.setup_rag_chain()
        else:
            self.vectorstore = None
            self.rag_chain = None

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_memory(self):
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        self.logger.info(f"Available system memory: {available_memory:.1f} GB")
        if available_memory < 4.0:
            self.logger.warning("Low memory detected! Consider optimizing your model execution.")

    def load_documents(self):
        if self.doc_path.endswith(".pdf"):
            loader = PyPDFLoader(self.doc_path)
        else:
            loader = TextLoader(self.doc_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=10)
        return text_splitter.split_documents(documents)

    def create_vectorstore(self):
        vectorstore = FAISS.from_documents(self.documents, self.embeddings)  # Usa FAISS invece di Chroma
        return vectorstore

    def setup_rag_chain(self):
        retriever = self.vectorstore.as_retriever()
        template = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question:
        If you don't know the answer, then do not answer from your own knowledge.
        Keep the answer concise.

        Question: {question}
        Context: {context}
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        return ( {"context": retriever, "question": RunnablePassthrough()} | prompt | self.llm | StrOutputParser() )

    def query(self, question):
        return self.rag_chain.invoke(question)

if __name__ == "__main__":
    doc_path = "pda_a.pdf"  # Update this with the actual path
    rag = CPUOptimizedRAGPipeline(doc_path=doc_path)
    while True:
        query = input("Enter Question: ")
        print(rag.query(query))
