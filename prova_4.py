from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
import logging
import psutil
import os

class RAGPipeline:
    def __init__(self, model_name: str = "llama3.2", max_memory_gb: float = 3.0):
        """Inizializza la pipeline RAG, controlla la memoria e carica i modelli."""
        self.setup_logging()  # ✅ Ora esiste prima di essere chiamato
        self.check_system_memory(max_memory_gb)

        # Carica il modello LLM
        self.llm = OllamaLLM(model=model_name)

        # Inizializza gli embedding usando Hugging Face
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Definisce il template per il prompt
        self.prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context. Be concise.
        If you cannot find the answer in the context, say "I cannot answer this based on the provided context."
        
        Context: {context}
        Question: {question}
        Answer: """)

    def setup_logging(self):
        """Configura il logging per il monitoraggio della memoria e delle operazioni."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_system_memory(self, max_memory_gb: float):
        """Controlla se il sistema ha abbastanza memoria disponibile."""
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        self.logger.info(f"Available system memory: {available_memory:.1f} GB")
        if available_memory < max_memory_gb:
            self.logger.warning("Memory is below recommended threshold.")

    def load_and_split_documents(self, file_path: str) -> List[Document]:
        """Carica e divide il documento in chunk."""
        loader = TextLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )
        splits = text_splitter.split_documents(documents)
        self.logger.info(f"Created {len(splits)} document chunks")
        return splits

    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """Crea il database vettoriale FAISS per la ricerca."""
        batch_size = 32
        vectorstore = FAISS.from_documents(documents[:batch_size], self.embeddings)
        
        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectorstore.add_documents(batch)
            self.logger.info(f"Processed batch {i//batch_size + 1}")
        return vectorstore

    def setup_rag_chain(self, vectorstore: FAISS):
        """Configura la catena RAG per il recupero e la generazione delle risposte."""
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2, "fetch_k": 3})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def query(self, chain, question: str) -> str:
        """Interroga il sistema RAG monitorando l'uso della memoria."""
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.logger.info(f"Memory usage: {memory_usage:.1f} MB")
        return chain.invoke(question)

# Funzione principale per eseguire la pipeline
def main():
    rag = RAGPipeline(model_name="llama3.2", max_memory_gb=4.0)

    # Caricamento e pre-processing dei documenti
    documents = rag.load_and_split_documents("gpu_market.txt")

    # Creazione del database vettoriale
    vectorstore = rag.create_vectorstore(documents)

    # Configurazione della catena RAG
    chain = rag.setup_rag_chain(vectorstore)

    # Eseguire una query di esempio
    question = "Make me a summary of 50 characters."
    response = rag.query(chain, question)
    print(f"Question: {question}\nAnswer: {response}")

# Avvio dello script
if __name__ == "__main__":
    main()
