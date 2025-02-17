import os
import logging
import psutil
import chromadb
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class CPUOptimizedRAGPipeline:
    def __init__(
        self,
        model_name="llama3.2",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        docs_folder="documents",
        vectorstore_type="faiss",
        use_chat_mode=False
    ):
        self.setup_logging()
        self.check_memory()
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.docs_folder = docs_folder
        self.vectorstore_type = vectorstore_type.lower()
        self.use_chat_mode = use_chat_mode

        # Usa Hugging Face embeddings invece di OllamaEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.llm = ChatOllama(model=self.model_name) if self.use_chat_mode else OllamaLLM(model=self.model_name)

        # Caricamento automatico documenti e creazione del vectorstore/chain
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
            chunks = text_splitter.split_documents(documents)
            self.logger.info(f"File '{file}': {len(chunks)} chunk estratti.")
            all_documents.extend(chunks)
        
        self.logger.info(f"Loaded {len(all_documents)} chunks from {len(os.listdir(self.docs_folder))} files.")
        return all_documents

    def create_vectorstore(self):
        if self.vectorstore_type == "faiss":
            faiss_path = "faiss_index"
            if os.path.exists(faiss_path):
                vectorstore = FAISS.load_local(faiss_path, self.embeddings, allow_dangerous_deserialization=True)
            else:
                vectorstore = FAISS.from_documents(self.documents, self.embeddings)
                vectorstore.save_local(faiss_path)
        elif self.vectorstore_type == "chromadb":
            chroma_client = chromadb.PersistentClient(path="chromadb_index")
            collection = chroma_client.get_or_create_collection(name="rag_collection")
            for i, doc in enumerate(self.documents):
                collection.add(documents=[doc.page_content], ids=[str(i)])
            vectorstore = collection
        else:
            raise ValueError("Unsupported vectorstore. Use 'faiss' or 'chromadb'.")
        return vectorstore

    def setup_rag_chain(self):
        if self.vectorstore_type == "faiss":
            retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 30})
        else:
            def chromadb_retriever(query_text):
                results = self.vectorstore.query(query_texts=[query_text], n_results=3)
                return results["documents"]
            retriever = chromadb_retriever
        
        template = """
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question:
        If you don't know the answer, then do not answer from your own knowledge.
        Keep the answer concise.
        
        #### Retrieved Context ####
        {context}
        
        #### Question ####
        {question}
        
        #### LLM Response ####
        """
        
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

    def export_chunks(self, output_folder="exported_chunks"):
        """
        Esporta i chunk di ogni documento in file di testo separati per ciascun file.
        Questo permette di verificare che tutti i chunk siano stati estratti correttamente.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file in os.listdir(self.docs_folder):
            file_path = os.path.join(self.docs_folder, file)
            
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                self.logger.info(f"File non supportato, saltato: {file}")
                continue

            try:
                documents = loader.load()
            except Exception as e:
                self.logger.error(f"Errore nel caricamento di {file}: {e}")
                continue

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)

            # Crea un file di output per il file corrente
            output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_chunks.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                for i, chunk in enumerate(chunks):
                    f.write(f"--- Chunk {i+1} ---\n")
                    # Si assume che il contenuto del chunk sia in 'page_content'
                    content = getattr(chunk, "page_content", str(chunk))
                    f.write(content + "\n\n")
            self.logger.info(f"Esportati {len(chunks)} chunk in {output_file}")

if __name__ == "__main__":
    vectorstore_choice = input("Choose vectorstore (faiss/chromadb): ").strip().lower()
    chat_mode_choice = input("Use chat mode? (yes/no): ").strip().lower() == "yes"
    
    rag = CPUOptimizedRAGPipeline(docs_folder="documents", vectorstore_type=vectorstore_choice, use_chat_mode=chat_mode_choice)
    
    mode = input("Scegli la modalità: (1) Q&A oppure (2) Esporta chunk: ").strip()
    if mode == "2":
        rag.export_chunks()
        print("Esportazione completata. Controlla la cartella 'exported_chunks'.")
    else:
        while True:
            query = input("Enter Question (or type 'exit' to quit): ")
            if query.lower() == "exit":
                break
            print("Answer:", rag.query(query))
