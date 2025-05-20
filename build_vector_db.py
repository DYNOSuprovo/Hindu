from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Step 1: Load all PDFs
loader = DirectoryLoader("pdfs", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Step 2: Convert to embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Step 3: Create and persist vector store
db = Chroma.from_documents(documents, embedding=embedding, persist_directory="db")
db.persist()
print("✅ Vector store created successfully from all PDFs.")