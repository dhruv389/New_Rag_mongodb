from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import logging
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ STEP 4: Connect to MongoDB
MONGO_URI = os.environ.get("MONGO_URI")  # Get from environment
if not MONGO_URI:
    raise ValueError("MONGO_URI not set in environment")
client = MongoClient(MONGO_URI)
db = client["test"]  # Replace "test" with your database name
collection = db["machindetails"]  # Replace "machindetails" with your collection name

# ✅ STEP 5: Load Documents from MongoDB
def load_documents():
    documents = []
    for doc in collection.find():
        text = "\n".join([f"{k}: {v}" for k, v in doc.items() if k != "_id"])
        documents.append(Document(page_content=text))
    return documents

# ✅ STEP 6: Build the RAG Chain with FAISS and Gemini
def build_rag_chain():
    docs = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Use the following context to answer the question:
    {context}

    Question: {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

rag_chain = build_rag_chain() # Build the chain

class QueryInput(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query/", response_model=QueryResponse)
async def query_rag(query_input: QueryInput):
    """
    Endpoint to query the RAG model.
    """
    try:
        logger.info(f"Received query: {query_input.query}")
        answer = rag_chain.invoke(query_input.query)
        logger.info(f"Generated answer: {answer}")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)