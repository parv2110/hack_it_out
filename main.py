import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import pipeline
from google.api_core.exceptions import ResourceExhausted

# Load environment variables
load_dotenv(dotenv_path="main.env")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
genai.configure(api_key=GENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Create vector DB path if not exists
VECTOR_DB_PATH = "faiss_db"
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize local HuggingFace summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Utility: split text into chunks
def split_text(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)

@app.get("/")
def home():
    return {"message": "Welcome to Gemini + HuggingFace Q&A API. Use /upload and /ask endpoints."}

@app.post("/upload")
async def upload_file(user_id: str, file: UploadFile = File(...)):
    try:
        file_path = f"temp_{user_id}.pdf"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        chunks = split_text(pages)

        documents = [Document(page_content=chunk.page_content) for chunk in chunks]
        vectors = FAISS.from_documents(documents, embedding_model)
        vectors.save_local(f"{VECTOR_DB_PATH}/{user_id}")

        os.remove(file_path)

        return {
            "message": f"Uploaded and indexed document for user '{user_id}'",
            "chunks": len(documents)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask")
async def ask_question(user_id: str, query: str):
    try:
        db_path = f"{VECTOR_DB_PATH}/{user_id}"
        if not os.path.exists(db_path):
            return JSONResponse(status_code=404, content={"error": "No document found for this user. Please upload first."})

        db = FAISS.load_local(
            db_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever()
        results = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content[:300] for doc in results[:2]])

        # Primary: Gemini
        try:
            model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
            response = model.generate_content([
                f"Context:\n{context}",
                f"Question:\n{query}"
            ])
            return {"answer": response.text.strip(), "model": "gemini"}
        
        except ResourceExhausted:
            # Fallback: HuggingFace summarizer
            summary = summarizer(context, max_length=256, min_length=30, do_sample=False)[0]['summary_text']
            return {"answer": summary.strip(), "model": "huggingface (fallback)"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/clear/{user_id}")
async def clear_user(user_id: str):
    try:
        shutil.rmtree(f"{VECTOR_DB_PATH}/{user_id}")
        return {"message": f"Cleared FAISS data for user '{user_id}'"}
    except FileNotFoundError:
        return {"error": "No data found for this user"}
