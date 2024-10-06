import pandas as pd
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Load CSV and filter Italian Q&A
def load_italian_qa(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df = df.drop_duplicates(subset=['question', 'answer', 'chunk'])
    return df[['question', 'answer', 'chunk', 'title', 'subheading', 'pdf_name', 'page_number']]

# Generate and normalize embeddings using SentenceTransformer with progress bar
def generate_embeddings(texts: list, model) -> np.ndarray:
    embeddings = []
    for text in tqdm(texts, desc="Generating embeddings"):
        embedding = np.array(model.encode(text))
        # Normalize the embedding to unit length
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    return np.array(embeddings)

# Create FAISS index and add embeddings
def create_faiss_index(data: pd.DataFrame, model, embedding_dim: int) -> FAISS:
    # Prepare documents with metadata and filter chunks
    documents = [
        Document(page_content=row['question'], metadata=row.to_dict())
        for _, row in data.iterrows() if len(row['chunk']) >= 20
    ]

    # Initialize InMemoryDocstore
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    # Initialize the FAISS index with cosine similarity
    faiss_index = faiss.IndexFlatIP(embedding_dim)

    # Generate and normalize embeddings
    embeddings = generate_embeddings([doc.page_content for doc in documents], model)
    faiss_index.add(embeddings)

    vector_store = FAISS(
        embedding_function=model,
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    
    return vector_store

# Save FAISS index
def save_faiss_index(vector_store: FAISS, index_path: str):
    vector_store.save_local(index_path)

# Load FAISS index
def load_faiss_index(index_path: str, model) -> FAISS:
    return FAISS.load_local(
        index_path, 
        embeddings=model,
        allow_dangerous_deserialization=True  # Enable this if you trust your data source
    )

# Perform semantic search
def perform_semantic_search(query: str, vector_store: FAISS, model) -> list:
    query_embedding = np.array(model.encode(query))
    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    results = vector_store.similarity_search_by_vector(query_embedding, k=3)
    return results

# Main function
def main():
    csv_file = 'new_output_qa_pairs.csv'
    index_path = 'faiss_index'

    # Load and process the CSV file
    data = load_italian_qa(csv_file)

    # Load the larger Italian language model from Hugging Face
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model = SentenceTransformer(model_name)

    # Set the embedding dimension to match the model output
    embedding_dim = model.get_sentence_embedding_dimension()

    # Create FAISS index
    vector_store = create_faiss_index(data, model, embedding_dim)

    # Save FAISS index
    save_faiss_index(vector_store, index_path)

    # Load FAISS index
    loaded_vector_store = load_faiss_index(index_path, model)

    # Perform semantic search
    query = "Qual è il tipo di corso di studio che può essere definito interfacoltà?"
    results = perform_semantic_search(query, loaded_vector_store, model)

    # Print the top 3 answers with metadata
    for result in results:
        print(f"Question: {result.page_content}")
        print(f"Answer: {result.metadata['answer']}")
        print(f"Metadata: {result.metadata}\n")
        print("-----------------")

if __name__ == "__main__":
    main()
