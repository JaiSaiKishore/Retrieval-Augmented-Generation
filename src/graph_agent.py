import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langgraph.graph import END, StateGraph
from typing import List, TypedDict

# 1. Load Secrets
load_dotenv()

# --- CONFIGURATION ---
DB_PATH = "vector_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# The "Judge" model that scores how relevant a document is (runs on your 3050)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" 

# --- 2. SETUP RESOURCES ---
print("⚙️ Loading Models & Database...")

# A. Embeddings (GPU)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cuda'}
)

# B. Vector DB
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# C. Hybrid Search Setup (The "Advanced" part)
# Chroma doesn't save BM25 indexes, so we rebuild it in 1 second from the data
print("⚙️ Building Hybrid Search Index...")
db_data = vector_db.get() # Fetch all data
doc_objects = [
    Document(page_content=t, metadata=m) 
    for t, m in zip(db_data['documents'], db_data['metadatas'])
]

bm25_retriever = BM25Retriever.from_documents(doc_objects)
bm25_retriever.k = 5

vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# The Ensemble: 50% Keyword Match, 50% Vector Meaning
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# D. Re-ranker Model (Local GPU)
reranker = CrossEncoder(RERANK_MODEL, device='cuda')

# E. LLM (Cloud - Groq)
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# --- 3. DEFINE AGENT STATE ---
class GraphState(TypedDict):
    question: str
    context: List[Document]
    answer: str

# --- 4. DEFINE AGENT ACTIONS (NODES) ---

def retrieve(state: GraphState):
    """Result of Hybrid Search"""
    print(f"🔍 Searching for: {state['question']}")
    documents = ensemble_retriever.invoke(state["question"])
    return {"context": documents}

def rerank(state: GraphState):
    """Re-ranks and filters documents using Cross-Encoder"""
    print("⚖️ AI Judge: Re-ranking documents...")
    question = state["question"]
    documents = state["context"]
    
    if not documents:
        return {"context": []}
    
    # Create pairs [Query, Text] for the AI to judge
    pairs = [[question, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    
    # Sort by score (Highest relevance first)
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    # Keep top 3 best chunks
    top_docs = [doc for doc, score in scored_docs[:3]]
    
    return {"context": top_docs}

def generate(state: GraphState):
    """Generates answer using Llama 3"""
    print("💡 Llama 3: Generating Answer...")
    question = state["question"]
    context = state["context"]
    
    prompt = ChatPromptTemplate.from_template(
        """You are a senior financial analyst. 
        Answer the question based ONLY on the following context. 
        If the context doesn't answer the question, say you don't know.
        
        Context:
        {context}
        
        Question: {question}
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
    return {"answer": response}

# --- 5. BUILD THE GRAPH ---
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("rerank", rerank)
workflow.add_node("generate", generate)

# Connect Edges (The Flow)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", END)

# Compile the brain
app_graph = workflow.compile()