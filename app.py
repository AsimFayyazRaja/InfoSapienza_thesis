from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from utils import load_faiss_index, perform_semantic_search
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI()

# Global variables for model and vector store
model = None
vector_store = None

# Load model and FAISS index during startup
@app.on_event("startup")
async def startup_event():
    global model, vector_store
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model = SentenceTransformer(model_name)
    index_path = 'faiss_index'
    vector_store = load_faiss_index(index_path, model)
    print("Model and FAISS index loaded successfully.")

# Response model for semantic search
class SearchResult(BaseModel):
    question: str
    answer: str
    similarity: float
    metadata: dict

# Define a route for semantic search using form data
@app.post("/search", response_model=list[SearchResult])
async def search(
    question: str = Form(..., description="The question to query."),
    num: int = Form(3, description="Number of results to return."),
    threshold: float = Form(0.5, description="Similarity threshold for filtering results."),
    top: bool = Form(True, description="If true, return top results regardless of threshold.")
):
    if num <= 0:
        raise HTTPException(status_code=400, detail="Number of results must be greater than zero")
    
    results = perform_semantic_search(question, vector_store, model, num, threshold, top)
    return [
        SearchResult(
            question=result.page_content,
            answer=result.metadata['answer'],
            similarity=result.metadata['similarity'],
            metadata=result.metadata
        )
        for result in results
    ]

# Health check route
@app.get("/")
async def root():
    return {"message": "Semantic search API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)