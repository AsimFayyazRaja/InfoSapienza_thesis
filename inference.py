import faiss
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import numpy as np

# Load FAISS index
def load_faiss_index(index_path: str, model) -> FAISS:
    return FAISS.load_local(
        index_path, 
        embeddings=model,
        allow_dangerous_deserialization=True  # Enable this if you trust your data source
    )

# Perform semantic search and return similarity scores
def perform_semantic_search(query: str, vector_store: FAISS, model) -> list:
    # Encode and normalize the query
    query_embedding = np.array(model.encode(query))
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Perform search
    distances, indices = vector_store.index.search(np.array([query_embedding]), k=3)
    
    # Calculate cosine similarity (since we used IP, the similarity is 1 - distance)
    similarities = distances[0]
    
    results = vector_store.similarity_search_by_vector(query_embedding, k=3)
    
    # Attach similarities to results
    for result, similarity in zip(results, similarities):
        result.metadata['similarity'] = similarity
    
    return results

# Main function
def main():
    index_path = 'faiss_index'

    # Load the model used for embeddings
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model = SentenceTransformer(model_name)

    # Load FAISS index
    loaded_vector_store = load_faiss_index(index_path, model)

    # Perform semantic search
    query = "Qual è il tipo di corso di studio che può essere definito interfacoltà?"
    results = perform_semantic_search(query, loaded_vector_store, model)

    # Print the top 3 answers with metadata
    for result in results:
        print(f"Question: {result.page_content}")
        print(f"Answer: {result.metadata['answer']}")
        print(f"Similarity: {result.metadata['similarity']:.4f}")
        print(f"Metadata: {result.metadata}\n")
        print("-----------------")

if __name__ == "__main__":
    main()
