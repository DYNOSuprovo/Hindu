import os
from langchain_groq.chat_models import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Set your API key
os.environ["GROQ_API_KEY"] = "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"

# Load Groq LLM
llm = ChatGroq(model_name="gemma2-9b-it", temperature=0.3)

# Load vector store
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="db", embedding_function=embedding)
retriever = db.as_retriever()

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# Ask questions
print("Ask questions to the Groq model (type 'exit' to quit):")
while True:
    question = input("Question: ")
    if question.lower() == 'exit':
        break
    try:
        response = qa_chain.invoke(question)  # using invoke instead of run
        print("Answer:", response)
    except Exception as e:
        print("Error:", str(e))
