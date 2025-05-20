from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema import Document  # Correct import here

# Step 1: Load all PDFs
loader = DirectoryLoader("pdfs", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Step 1.5: Ensure all documents have string page_content only
def clean_doc(doc):
    content = doc.page_content
    if not isinstance(content, str):
        # Convert dict or others to string safely
        return Document(page_content=str(content), metadata=doc.metadata)
    return doc

cleaned_docs = [clean_doc(doc) for doc in documents]

# Step 2: Convert to embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 3: Create and persist vector store
db = Chroma.from_documents(cleaned_docs, embedding=embedding, persist_directory="db")
db.persist()

print("✅ Vector store created successfully from all PDFs.")
