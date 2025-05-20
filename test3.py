from pdfs.fetch import download_and_extract
# Google Drive File ID
DB_FILE_ID = "1pvrwMfMQ4oPYU-IhHN3mTurM5JJBHCVx"
# Automatically download and extract if missing
download_and_extract(DB_FILE_ID, "db.zip", "db")

# IMPORTANT: Ensure pysqlite3 is imported and takes precedence over system sqlite3
# This must be done BEFORE any other imports that might implicitly use sqlite3 (like chromadb)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import requests
from dotenv import load_dotenv
# Corrected imports from langchain_community
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, create_history_aware_retriever # Added create_history_aware_retriever
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from concurrent.futures import ThreadPoolExecutor
# sentence_transformers is used for the underlying model, keep this import
from sentence_transformers import SentenceTransformer
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
# Import message types for history reconstruction
from langchain_core.messages import HumanMessage, AIMessage, MessagesPlaceholder # Added MessagesPlaceholder

import google.generativeai as genai
import logging # Add logging for better error inspection
import string # Import string module for punctuation removal
import re # Import regex for regional preference extraction

# Import chromadb directly for explicit client configuration
import chromadb
from chromadb.config import Settings # Import Settings for explicit configuration

# --- Streamlit Page Configuration ---
# This MUST be the first Streamlit command in your script.
st.set_page_config(page_title="🕉️ Hindu Scripture Advisor", layout="wide")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Application started.")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- API Key Checks and Configuration ---
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
    logging.error("GEMINI_API_KEY not found.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

try:
    genai.configure(api_key=GEMINI_API_KEY)
    llm_gemini = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.5)
    logging.info("Google Generative AI configured successfully.")
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {e}")
    logging.error(f"Google GenAI Configuration Error: {e}")
    st.stop()

if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY not found. Groq suggestions will be unavailable.")
    logging.warning("GROQ_API_KEY not found.")

try:
    logging.info("Starting HuggingFaceEmbeddings initialization with 'all-MiniLM-L6-v2' on CPU.")

    st.info("Loading embedding model (this may take a moment)...")
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", # Pass the model name as a string
        model_kwargs={'device': 'cpu'}, # Specify the device here
        encode_kwargs={'normalize_embeddings': True}
    )
    logging.info("HuggingFaceEmbeddings initialized successfully.")

    chroma_db_directory = "db"
    if not os.path.exists(chroma_db_directory):
        error_msg = (
            f"ChromaDB directory '{chroma_db_directory}' not found. "
            "Please ensure the DB is initialized first with your Hindu scriptures."
        )
        st.error(error_msg)
        logging.error(error_msg)
        st.stop()

    # --- NEW: Explicitly configure and initialize Chroma Client ---
    # Define ChromaDB settings to use Disk persistence and the correct path
    # This also implicitly handles the sqlite3 version by using the pysqlite3-binary
    chroma_settings = Settings(
        persist_directory=chroma_db_directory,
        is_persistent=True # Explicitly state it's persistent
    )
    # Create a Chroma client with the defined settings
    # This client will then be passed to the Langchain Chroma wrapper
    chroma_client = chromadb.Client(settings=chroma_settings)
    logging.info("Chroma Client initialized successfully with explicit settings.")

    # Initialize Langchain's Chroma vector store with the explicit client
    # We pass the client directly, so Langchain's Chroma doesn't try to create its own
    db = Chroma(
        client=chroma_client, # Pass the explicitly created client
        embedding_function=embedding,
        collection_name="hindu_scriptures" # It's good practice to name your collection
    )
    logging.info("Chroma DB loaded successfully.")
    st.success("Vector database loaded successfully.")

except Exception as e:
    error_msg = f"VectorDB setup error: {str(e)}"
    st.error(error_msg)
    logging.exception("Full VectorDB setup traceback:")
    st.stop()

# --- RAG Prompt Template (UPDATED for Hindu Scriptures) ---
scripture_prompt = PromptTemplate.from_template("""
You are an AI assistant specialized in Hindu scriptures and spiritual guidance.
Based on the following conversation history and the user's query, provide a simple, practical, and culturally relevant answer or guidance.
If the user's query is vague or refers to a previous topic (e.g., "he", "it"), **always use the chat history to understand who or what is being referred to.**
If a specific **{spiritual_concept}** or **{life_problem}** is mentioned or inferred, prioritize insights from relevant **{scripture_source}**.
Focus on teachings from primary Hindu texts and their practical application.
Be helpful, encouraging, and specific where possible.
Use the chat history to understand the context of the user's current query and maintain continuity.
Strictly adhere to the **{spiritual_concept}** and **{life_problem}** requirements, and the **{scripture_source}** preference if specified.
**Ensure your language is simple, clear, and easy for a general audience to understand.**

Chat History:
{chat_history}

Context from Knowledge Base:
{context}

User Query:
{question}

Guidance based on Hindu Scripture (Tailored for {spiritual_concept}, {life_problem}, from {scripture_source}):
""")
logging.info("RAG Prompt template created for Hindu scriptures.")

# --- Retrieval QA Chain Setup ---
# Define a prompt for the history-aware retriever to rephrase the question
rephrase_prompt = PromptTemplate.from_template("""
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")

# Create the history-aware retriever
# This will take chat history and the current question to produce a standalone question for retrieval
history_aware_retriever = create_history_aware_retriever(
    llm_gemini,
    db.as_retriever(search_kwargs={"k": 5}), # Use the base retriever here
    rephrase_prompt
)
logging.info("History-aware retriever initialized.")

# Now, the RAG chain will use the history-aware retriever
# The prompt for the RAG chain will still include chat history for context in generation
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_gemini,
        # The retriever here is the history-aware one
        retriever=history_aware_retriever,
        chain_type="stuff",
        return_source_documents=True,
        # Pass the scripture_prompt here for the generation step
        chain_type_kwargs={"prompt": scripture_prompt}
    )
    logging.info("Retrieval QA Chain initialized successfully with history-aware retriever.")
except Exception as e:
    st.error(f"QA Chain setup error: {e}")
    logging.exception("Full QA Chain setup traceback:")
    st.stop()


# --- Session History Management (No Change Needed) ---
store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        logging.info(f"Creating new Langchain session history in 'store' for: {session_id}")
        store[session_id] = ChatMessageHistory()
    else:
        logging.info(f"Retrieving existing Langchain session history from 'store' for: {session_id}")
    return store[session_id]

# --- Conversational QA Chain Setup (Uses new prompt) ---
# The RunnableWithMessageHistory now wraps the qa_chain which internally uses the history-aware retriever
conversational_qa_chain = RunnableWithMessageHistory(
    qa_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
    output_messages_key="answer"
)
logging.info("Conversational QA Chain initialized.")

# --- Merge Prompt Templates (UPDATED for Hindu Scriptures) ---
merge_prompt_template_default = """
You are a Hindu scripture and spiritual guidance assistant.
Your goal is to synthesize information from a primary RAG-based answer and several other AI suggestions into a single, coherent, and practical guidance or answer for **{life_problem}**, referencing **{spiritual_concept}** from **{scripture_source}** if specified.
Prioritize the Primary RAG Answer. If it's weak or irrelevant, use Additional Suggestions.
Ensure the final guidance is clear, actionable, and respectful of Hindu traditions. Present as a clear paragraph or a list of points.
If the user's input was *only* a greeting, respond politely. For inputs that include a greeting but also contain a query, focus on answering the query.
**Ensure your language is simple, clear, and easy for a general audience to understand.**

Primary RAG Answer:
{rag}

Additional Suggestions:
- LLaMA Suggestion: {llama}
- Mixtral Suggestion: {mixtral}
- Gemma Suggestion: {gemma}

Refined and Merged Guidance (Tailored for {spiritual_concept}, {life_problem}, from {scripture_source}):
"""
merge_prompt_default = PromptTemplate.from_template(merge_prompt_template_default)

merge_prompt_template_table = """
You are a Hindu scripture and spiritual guidance assistant.
Your goal is to synthesize information from a primary RAG-based answer and several other AI suggestions into a single, coherent, and practical guidance or answer for **{life_problem}**, referencing **{spiritual_concept}** from **{scripture_source}** if specified.
Prioritize the Primary RAG Answer. If it's weak or irrelevant, use Additional Suggestions.
Ensure the final guidance is clear, actionable, and respectful of Hindu traditions.
**You MUST present the final guidance as a clear markdown table if appropriate. Include columns for Concept/Teaching, Scripture Reference, and Practical Application.**
If the user's input was *only* a greeting, respond politely. For inputs that include a greeting but also contain a query, focus on answering the query.
**Ensure your language is simple, clear, and easy for a general audience to understand.**

Primary RAG Answer:
{rag}

Additional Suggestions:
- LLaMA Suggestion: {llama}
- Mixtral Suggestion: {mixtral}
- Gemma Suggestion: {gemma}

Refined and Merged Guidance (Tailored for {spiritual_concept}, {life_problem}, from {scripture_source}, in markdown table format if applicable):
"""
merge_prompt_table = PromptTemplate.from_template(merge_prompt_template_table)
logging.info("Merge Prompt templates created for Hindu scriptures.")

def groq_scripture_answer(model_name: str, query: str, spiritual_concept: str = "general", life_problem: str = "guidance", scripture_source: str = "Hindu scriptures") -> str:
    if not GROQ_API_KEY:
        return f"Groq API key not available."
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        groq_model_map = {"llama": "llama3-70b-8192", "mixtral": "mixtral-8x7b-32976", "gemma": "gemma2-9b-it"}
        actual_model_name = groq_model_map.get(model_name.lower(), model_name)
        # Replaced f-string with triple quotes and removed .replace() for robustness
        prompt_content = f"""User query: '{query}'. Provide a concise, practical spiritual guidance or answer related to **{spiritual_concept}** for **{life_problem}**, referencing **{scripture_source}** if applicable. Be brief and use simple, easy-to-understand English."""
        payload = {"model": actual_model_name, "messages": [{"role": "user", "content": prompt_content}], "temperature": 0.5, "max_tokens": 250}
        logging.info(f"Calling Groq API: {actual_model_name} for query: '{query}' (Concept: {spiritual_concept}, Problem: {life_problem}, Source: {scripture_source})")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data and data.get('choices') and data['choices'][0].get('message'):
            return data['choices'][0]['message']['content']
        return f"No suggestion from {actual_model_name} (empty/malformed response)."
    except requests.exceptions.Timeout: return f"Timeout error from {model_name}."
    except requests.exceptions.RequestException as e: return f"Request error from {model_name}: {e}"
    except Exception as e: return f"Error from {model_name}: {e}"

@st.cache_data(show_spinner=False, ttl=3600)
def cached_groq_answers(query: str, spiritual_concept: str, life_problem: str, scripture_source: str) -> dict:
    logging.info(f"Fetching cached Groq answers for query: '{query}', concept: '{spiritual_concept}', problem: '{life_problem}', source: '{scripture_source}'")
    models = ["llama", "mixtral", "gemma"]
    results = {}
    if not GROQ_API_KEY: return {k: "Groq API key not available." for k in models}
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_model = {executor.submit(groq_scripture_answer, name, query, spiritual_concept, life_problem, scripture_source): name for name in models}
        for future in future_to_model:
            model_name = future_to_model[future]
            try: results[model_name] = future.result()
            except Exception as e: results[model_name] = f"Failed: {e}"
    return results

# --- Helper Functions (UPDATED for Hindu Scriptures) ---
GREETINGS = ["hi", "hello", "hey", "namaste", "yo", "pranam", "jai shree ram", "om namah shivaya", "radhe radhe", "good morning", "good afternoon", "good evening"]
TASK_KEYWORDS = [
    "dharma", "karma", "moksha", "atman", "brahman", "yoga", "meditation", "bhakti", "jnana", "seva",
    "stress", "anxiety", "fear", "sadness", "anger", "grief", "purpose", "meaning of life", "suffering",
    "bhagavad gita", "veda", "upanishad", "purana", "ramayana", "mahabharata", "scripture", "text", "shastra",
    "guidance", "solution", "advice", "teachings", "principles", "philosophy", "answer", "explain", "meaning",
    "table", "format", "chart", "show", "give", "list", "bullet", "points", "itemize", "enumerate"
]
FORMATTING_KEYWORDS = ["table", "tabular", "chart", "format", "list", "bullet", "points", "itemize", "enumerate"]

def is_greeting(query: str) -> bool:
    if not query: return False
    cleaned_query = query.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    contains_task_keyword = any(keyword in cleaned_query for keyword in TASK_KEYWORDS)
    is_short_greeting = cleaned_query in GREETINGS and len(cleaned_query.split()) <= 3
    return is_short_greeting and not contains_task_keyword

def is_formatting_request(query: str) -> bool:
    """Checks if the query is primarily a formatting request."""
    if not query: return False
    cleaned_query = query.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    words = cleaned_query.split()

    if not any(keyword in cleaned_query for keyword in FORMATTING_KEYWORDS):
        return False # No formatting keywords found

    if len(words) <= 5:
        non_formatting_words = 0
        for word in words:
            if word not in FORMATTING_KEYWORDS and word not in ["in", "a", "as", "give", "me", "show", "it", "that", "please", "can", "you"]:
                non_formatting_words +=1
        if non_formatting_words <= 1:
            logging.info(f"Query '{query}' identified as formatting request (short, specific keywords).")
            return True
    only_formatting_and_fillers = True
    for word in words:
        if word not in FORMATTING_KEYWORDS and word not in ["in", "a", "as", "give", "me", "show", "it", "that", "please", "can", "you", "the", "for", "my", "me"]:
            only_formatting_and_fillers = False
            break
    if only_formatting_and_fillers:
        logging.info(f"Query '{query}' identified as formatting request (only formatting/fillers).")
        return True
    return False

def extract_spiritual_concept(query: str) -> str:
    query_lower = query.lower()
    concepts = {
        "dharma": ["dharma", "duty", "righteousness"],
        "karma": ["karma", "action", "consequence"],
        "moksha": ["moksha", "liberation", "salvation"],
        "atman": ["atman", "soul", "self"],
        "brahman": ["brahman", "ultimate reality"],
        "yoga": ["yoga", "meditation", "union"],
        "bhakti": ["bhakti", "devotion"],
        "jnana": ["jnana", "knowledge", "wisdom"],
        "seva": ["seva", "selfless service"],
        "reincarnation": ["reincarnation", "rebirth", "samsara"],
        "maya": ["maya", "illusion"],
        "nirvana": ["nirvana", "enlightenment"] # Though more Buddhist, sometimes used in broader spiritual context
    }
    for concept, keywords in concepts.items():
        if any(k in query_lower for k in keywords):
            return concept
    return "general" # Default for general spiritual concepts

def extract_life_problem(query: str) -> str:
    query_lower = query.lower()
    problems = {
        "stress": ["stress", "tension", "anxiety", "worry"],
        "anger": ["anger", "frustration", "irritation"],
        "grief": ["grief", "loss", "sadness", "sorrow"],
        "purpose": ["purpose", "meaning of life", "direction"],
        "fear": ["fear", "insecurity", "doubt"],
        "relationships": ["relationship", "family", "friends", "love"],
        "suffering": ["suffering", "pain", "hardship"]
    }
    for problem, keywords in problems.items():
        if any(k in query_lower for k in keywords):
            return problem
    return "guidance" # Default for general guidance

def extract_scripture_source(query: str) -> str:
    query_lower = query.lower()
    sources = {
        "bhagavad gita": ["bhagavad gita", "gita"],
        "veda": ["veda", "vedas", "rigveda", "yajurveda", "samaveda", "atharvaveda"],
        "upanishad": ["upanishad", "upanishads"],
        "purana": ["purana", "puranas"],
        "ramayana": ["ramayana", "ramayan"],
        "mahabharata": ["mahabharata", "mahabharat"],
        "yoga sutras": ["yoga sutras", "patanjali"],
        "dharma shastras": ["dharma shastras", "manu smriti"]
    }
    for source, keywords in sources.items():
        if any(k in query_lower for k in keywords):
            return " ".join([word.capitalize() for word in source.split()])
    return "Hindu scriptures" # Default

def contains_table_request(query: str) -> bool:
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ["table", "tabular", "chart", "in a table", "in table format", "as a table"])

# --- Streamlit UI (UPDATED for Hindu Scriptures) ---
# st.set_page_config(page_title="🕉️ Hindu Scripture Advisor", layout="wide") # Moved to top
with st.sidebar:
    st.markdown("👤 **Created by Lord d'Artagnan**")
    st.markdown("---")
    use_llms_toggle = st.toggle("🔄 Include expanded suggestions", value=True, help="Fetch from LLaMA, Mixtral, Gemma via Groq.")

st.title("🌺 Personalized Hindu Scripture Guidance")
st.markdown("Ask anything related to Hindu scriptures, spiritual concepts, or life problems for guidance.")

# --- Session State Initialization ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = "session_" + os.urandom(8).hex()
    st.session_state.messages = []
    st.session_state.spiritual_concept = "general" # Changed from dietary_type
    st.session_state.life_problem = "guidance"      # Changed from diet_goal
    st.session_state.scripture_source = "Hindu scriptures" # Changed from region
    st.session_state.table_format_requested = False
    st.session_state.last_substantive_query = ""
    get_session_history(st.session_state.session_id)
    logging.info(f"New Streamlit session: {st.session_state.session_id}")
else:
    get_session_history(st.session_state.session_id)
    if 'spiritual_concept' not in st.session_state: st.session_state.spiritual_concept = "general"
    if 'life_problem' not in st.session_state: st.session_state.life_problem = "guidance"
    if 'scripture_source' not in st.session_state: st.session_state.scripture_source = "Hindu scriptures"
    if 'table_format_requested' not in st.session_state: st.session_state.table_format_requested = False
    if 'last_substantive_query' not in st.session_state: st.session_state.last_substantive_query = ""
    logging.info(f"Existing session: {st.session_state.session_id}. Concept: {st.session_state.spiritual_concept}, Problem: {st.session_state.life_problem}, Source: {st.session_state.scripture_source}, Table: {st.session_state.table_format_requested}, LastQuery: {st.session_state.last_substantive_query}")

session_id_input = st.text_input("Session ID:", value=st.session_state.session_id, help="Change to switch conversation.")

if session_id_input and session_id_input != st.session_state.session_id:
    old_session_id = st.session_state.session_id
    st.session_state.session_id = session_id_input
    logging.info(f"User changed session ID from {old_session_id} to: {st.session_state.session_id}")
    current_langchain_history = get_session_history(st.session_state.session_id)
    new_ui_messages = []
    # Reset session specifics for the new/switched session
    st.session_state.spiritual_concept = "general"
    st.session_state.life_problem = "guidance"
    st.session_state.scripture_source = "Hindu scriptures"
    st.session_state.table_format_requested = False
    st.session_state.last_substantive_query = ""

    temp_last_substantive_query = ""
    for message_obj in current_langchain_history.messages:
        if isinstance(message_obj, HumanMessage):
            new_ui_messages.append({"role": "user", "content": message_obj.content})
            # Infer states from historical messages
            hist_concept = extract_spiritual_concept(message_obj.content)
            if hist_concept != "general": st.session_state.spiritual_concept = hist_concept
            hist_problem = extract_life_problem(message_obj.content)
            if hist_problem != "guidance": st.session_state.life_problem = hist_problem
            hist_source = extract_scripture_source(message_obj.content)
            if hist_source != "Hindu scriptures": st.session_state.scripture_source = hist_source
            if contains_table_request(message_obj.content): st.session_state.table_format_requested = True

            if not is_formatting_request(message_obj.content) and not is_greeting(message_obj.content):
                temp_last_substantive_query = message_obj.content
        elif isinstance(message_obj, AIMessage):
            new_ui_messages.append({"role": "assistant", "content": message_obj.content})
    st.session_state.last_substantive_query = temp_last_substantive_query
    st.session_state.messages = new_ui_messages
    logging.info(f"Switched to session {st.session_state.session_id}. Loaded {len(new_ui_messages)} UI messages. Last substantive: '{temp_last_substantive_query}'")
    st.toast(f"Switched to session: {st.session_id}. History loaded.")
    st.rerun()

if "messages" not in st.session_state: st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Enter your question or problem:", key="user_query_input")

if query:
    logging.info(f"Input: '{query}' for session: {st.session_state.session_id}")
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    pure_greeting = is_greeting(query)
    if pure_greeting:
        logging.info("Query is a pure greeting.")
        greeting_response = "Om Namah Shivaya! I am your Hindu Scripture Advisor. How may I help you today?"
        st.session_state.messages.append({"role": "assistant", "content": greeting_response})
        with st.chat_message("assistant"): st.markdown(greeting_response)
        get_session_history(st.session_state.session_id).add_ai_message(greeting_response)
    else:
        logging.info("Query is NOT a pure greeting. Processing...")
        with st.chat_message("assistant"):
            with st.spinner("Reflecting on the scriptures..."):
                # Determine the query to use for RAG and Groq
                query_for_rag_and_groq = query
                if is_formatting_request(query) and st.session_state.last_substantive_query:
                    query_for_rag_and_groq = st.session_state.last_substantive_query
                    logging.info(f"Identified formatting request. Using last substantive query for RAG/Groq: '{query_for_rag_and_groq}'")
                else:
                    st.session_state.last_substantive_query = query
                    logging.info(f"This is a substantive query. Updating last_substantive_query to: '{query}'")

                # Extract preferences from the *current* user query
                current_spiritual_concept = extract_spiritual_concept(query)
                current_life_problem = extract_life_problem(query)
                current_scripture_source = extract_scripture_source(query)
                current_table_request = contains_table_request(query)

                # Update session state based on *current* query's extractions
                if current_spiritual_concept != "general": st.session_state.spiritual_concept = current_spiritual_concept
                if current_life_problem != "guidance": st.session_state.life_problem = current_life_problem
                if current_scripture_source != "Hindu scriptures": st.session_state.scripture_source = current_scripture_source
                if current_table_request: st.session_state.table_format_requested = True

                logging.info(f"Processing with: Query for RAG/Groq='{query_for_rag_and_groq}', SessionConcept='{st.session_state.spiritual_concept}', SessionProblem='{st.session_state.life_problem}', SessionSource='{st.session_state.scripture_source}', TableRequested='{st.session_state.table_format_requested}'")

                rag_answer = "Could not retrieve from knowledge base."
                try:
                    # The conversational_qa_chain now internally uses the history-aware retriever
                    rag_result = conversational_qa_chain.invoke(
                        {"question": query_for_rag_and_groq,
                         "spiritual_concept": st.session_state.spiritual_concept,
                         "life_problem": st.session_state.life_problem,
                         "scripture_source": st.session_state.scripture_source},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                    if isinstance(rag_result, dict) and "answer" in rag_result:
                        rag_answer = rag_result["answer"]
                        logging.info(f"RAG answer: '{rag_answer[:100]}...'")
                        if "source_documents" in rag_result: logging.info(f"Retrieved {len(rag_result['source_documents'])} sources.")
                    else:
                        logging.warning(f"RAG result unexpected format: {rag_result}")
                except Exception as e:
                    rag_answer = f"Error in RAG: {e}"
                    logging.exception("RAG chain error:")

                groq_suggestions = {"llama": "N/A", "mixtral": "N/A", "gemma": "N/A"}
                if use_llms_toggle and GROQ_API_KEY:
                    groq_suggestions = cached_groq_answers(query_for_rag_and_groq, st.session_state.spiritual_concept, st.session_state.life_problem, st.session_state.scripture_source)
                elif not GROQ_API_KEY:
                    groq_suggestions = {k: "Groq API key not available." for k in groq_suggestions}

                final_answer = ""
                try:
                    current_merge_prompt = merge_prompt_table if st.session_state.table_format_requested else merge_prompt_default
                    logging.info(f"Using {'TABLE' if st.session_state.table_format_requested else 'DEFAULT'} merge prompt.")

                    merge_input = {
                        "rag": rag_answer,
                        "llama": groq_suggestions.get("llama", "N/A"),
                        "mixtral": groq_suggestions.get("mixtral", "N/A"),
                        "gemma": groq_suggestions.get("gemma", "N/A"),
                        "spiritual_concept": st.session_state.spiritual_concept,
                        "life_problem": st.session_state.life_problem,
                        "scripture_source": st.session_state.scripture_source
                    }
                    merge_chain = current_merge_prompt | llm_gemini
                    final_answer_obj = merge_chain.invoke(merge_input)
                    final_answer = final_answer_obj.content if hasattr(final_answer_obj, 'content') else str(final_answer_obj)
                    logging.info(f"Merged answer: '{final_answer[:100]}...'")
                except Exception as e:
                    final_answer = f"Error merging suggestions: {e}"
                    logging.exception("Merge process error:")

                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

                # Update Langchain history with the final merged answer
                session_history = get_session_history(st.session_state.session_id)
                if session_history.messages and isinstance(session_history.messages[-1], AIMessage):
                    logging.info("Popping last RAG AI message from Langchain history to replace with final merged answer.")
                    session_history.messages.pop()
                session_history.add_ai_message(final_answer)
                logging.info("Final merged answer added to Langchain history.")
