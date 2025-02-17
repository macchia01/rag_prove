import os
import logging
import psutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class CPUOptimizedRAGPipeline:
    def __init__(self, model_name="llama3.2", embedding_model="sentence-transformers/all-MiniLM-L6-v2", docs_folder="documents"):
        self.setup_logging()
        self.check_memory()
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.docs_folder = docs_folder

        # Usa Hugging Face embeddings invece di OllamaEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.llm = ChatOllama(model=self.model_name)

        # Caricamento automatico documenti
        self.documents = self.load_documents()
        self.vectorstore = self.create_vectorstore()
        self.rag_chain = self.setup_rag_chain()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_memory(self):
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        self.logger.info(f"Available system memory: {available_memory:.1f} GB")
        if available_memory < 4.0:
            self.logger.warning("Low memory detected! Consider optimizing your model execution.")

    def load_documents(self):
        all_documents = []
        for file in os.listdir(self.docs_folder):
            file_path = os.path.join(self.docs_folder, file)
            
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                continue  # Ignora file non supportati

            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            all_documents.extend(text_splitter.split_documents(documents))
        
        self.logger.info(f"Loaded {len(all_documents)} chunks from {len(os.listdir(self.docs_folder))} files.")
        return all_documents

    def create_vectorstore(self):
        vectorstore = FAISS.from_documents(self.documents, self.embeddings)  # Usa FAISS invece di ChromaDB
        return vectorstore

    def setup_rag_chain(self):
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        template = """You are an assistant for question-answering tasks.\n\n"""
        template += """Use the following retrieved context to answer the question:\n\n"""
        template += """Context: {context}\n\nQuestion: {question}\nAnswer: """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        return (
            {"context": retriever, "question": RunnablePassthrough()} 
            | prompt 
            | self.llm 
            | StrOutputParser()
        )

    def query(self, question):
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.logger.info(f"Memory usage: {memory_usage:.1f} MB")
        return self.rag_chain.invoke(question)

if __name__ == "__main__":
    rag = CPUOptimizedRAGPipeline(docs_folder="documents")
    
    while True:
        query = input("Enter Question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        print(rag.query(query))
