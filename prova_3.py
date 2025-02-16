from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Load the document
loader = PyPDFLoader("Foundations of LLMs.pdf")
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
docs = text_splitter.split_documents(documents=documents)

# ✅ Impostiamo il modello per funzionare sulla CPU
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}  # 🔴 Cambia "cuda" in "cpu"

# ✅ Caricamento del modello
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs
)

# ✅ Creazione dello store vettoriale FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# ✅ Salvataggio e caricamento dello store vettoriale
vectorstore.save_local("faiss_index_")
persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)

# ✅ Creazione del retriever
retriever = persisted_vectorstore.as_retriever()

# Initialize the LLaMA model
llm = OllamaLLM(model="llama3.2")

# Test with a sample prompt
response = llm.invoke("Tell me a joke")
print(response)

# ✅ Crea il modello di QA con il retriever
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# ✅ Loop interattivo per interrogare il sistema
while True:
    query = input("Type your query (or type 'Exit' to quit): \n")
    if query.lower() == "exit":
        break
    
    # 🔴 Sostituire .run() con .invoke()
    result = qa.invoke(query)  # ✅ Metodo corretto
    print(result)