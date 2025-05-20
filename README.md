🕉️ Hindu Scripture Advisor
This is a personalized AI chatbot designed to provide guidance and answers based on Hindu scriptures and spiritual concepts. It leverages Retrieval-Augmented Generation (RAG) to fetch relevant insights from a knowledge base of Hindu texts, combined with conversational history and suggestions from multiple Large Language Models (LLMs) for comprehensive and nuanced responses.

✨ Features
Retrieval-Augmented Generation (RAG): Fetches relevant passages from a dedicated vector database of Hindu scriptures to ground responses.

Conversational Memory: Maintains chat history to provide context-aware and continuous guidance.

Multi-LLM Integration: Combines insights from Google's Gemini-1.5-Flash with additional suggestions from LLaMA, Mixtral, and Gemma (via Groq API) for richer answers.

Dynamic Guidance: Extracts spiritual concepts (e.g., Dharma, Karma), life problems (e.g., stress, grief), and specific scripture sources (e.g., Bhagavad Gita, Vedas) from user queries to tailor responses.

Flexible Formatting: Can provide guidance in a narrative format or a structured markdown table (e.g., Concept, Scripture Reference, Practical Application).

Beautiful & Intuitive UI: Features a warm, inviting orangish gradient theme for a pleasant user experience.

🚀 Setup and Installation
Follow these steps to get the Hindu Scripture Advisor up and running on your local machine.

Prerequisites
Python 3.8+

pip (Python package installer)

Access to Google Gemini API and optionally Groq API.

1. Clone the Repository
git clone <your-repository-url>
cd <your-repository-name>

2. Create a Virtual Environment (Recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install Dependencies
Install the required Python libraries:

pip install -r requirements.txt

requirements.txt content:

streamlit
python-dotenv
langchain-community
langchain-google-genai
sentence-transformers
requests
protobuf==3.20.3 # Important: Pin protobuf to avoid compatibility issues

4. Set Up API Keys
Create a .env file in the root directory of your project (same level as test3.py or your main app file) and add your API keys:

GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
GROQ_API_KEY="YOUR_GROQ_API_KEY" # Optional, but recommended for expanded suggestions

Google Gemini API Key: Obtain from Google AI Studio.

Groq API Key: Obtain from Groq Console. If you don't provide this, the "expanded suggestions" feature will be unavailable.

5. Prepare Your Hindu Scripture Vector Database
This is a crucial step. The RAG component relies on a pre-built vector database of your Hindu scriptures.

Gather your data: Collect your Hindu scripture texts (e.g., .txt, .pdf files).

Create an ingestion script: You'll need a separate Python script to:

Load your scripture texts.

Split them into manageable chunks.

Generate embeddings for these chunks using HuggingFaceEmbeddings.

Store these embeddings in a ChromaDB instance.

Ensure the database is saved to the directory specified in the app: db_hindu_scriptures.

Example (Conceptual) Ingestion Script Snippet:

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Your Hindu scripture texts directory
DATA_PATH = "path/to/your/hindu_scripture_texts"
CHROMA_DB_DIR = "db_hindu_scriptures"

# Load documents (example for text files)
documents = []
for filename in os.listdir(DATA_PATH):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(DATA_PATH, filename))
        documents.extend(loader.load())

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={'normalize_embeddings': True}
)

# Create and persist the ChromaDB
print(f"Creating ChromaDB at {CHROMA_DB_DIR}...")
db = Chroma.from_documents(
    texts,
    embedding_model,
    persist_directory=CHROMA_DB_DIR
)
db.persist()
print("ChromaDB created and persisted successfully!")

Run this script once to create your db_hindu_scriptures directory.

🏃 How to Run
Once you have installed the dependencies and prepared your .env file and ChromaDB, run the Streamlit application:

streamlit run your_app_file_name.py # Replace 'your_app_file_name.py' with the actual name of your Python script

This will open the application in your web browser.

⚠️ Disclaimer
This advisor provides general spiritual guidance based on Hindu scriptures. It is an AI model and should not be considered a substitute for personal spiritual practice, the advice of qualified spiritual teachers, or professional counseling for complex life issues. Always exercise discernment and consult appropriate experts for personalized guidance.

👤 Created by Lord d'Artagnan
