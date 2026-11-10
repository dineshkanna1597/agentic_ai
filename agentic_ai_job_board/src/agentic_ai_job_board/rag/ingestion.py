from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from agentic_ai_job_board.gateway.router import get_embedding_model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_PATH = os.path.join(BASE_DIR, "qdrant_local_storage")

client = QdrantClient(path=STORAGE_PATH)
collection_name = 'career_portal'

VECTOR_SIZE = 4096

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

embeddings = get_embedding_model()

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

def load_document(file_path:str):
    '''Automatically selects the correct loader based on file extension'''

    if file_path.endswith('.pdf'):
        return PyMuPDFLoader(file_path).load()

    elif file_path.endswith('.docx'):
        return Docx2txtLoader(file_path).load()

    elif file_path.endswith('.txt'):
        return TextLoader(file_path, encoding="utf-8").load()

    else:
        return ValueError(f"Unsupported file format: {file_path}")


def ingest_directory(directory_path: str, doc_type: str):
    '''Walks through a directory, parses files, chunks them, and adds them to Qdrant'''

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
        )

    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            try:
                raw_docs = load_document(file_path)

                for doc in raw_docs:
                    doc.metadata.update(
                        {
                            'source':file,
                            'type':doc_type
                        }
                    )
                chunks = text_splitter.split_documents(raw_docs)
                vector_store.add_documents(chunks)
                print(f"Successfully processed and indexed: {file_path}")

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    if os.path.exists("src/agentic_ai_job_board/rag/data/resumes"):
        print("Ingesting Resumes...")
        ingest_directory("src/agentic_ai_job_board/rag/data/resumes", doc_type="resume")
        
    if os.path.exists("src/agentic_ai_job_board/rag/data/postings"):
        print("Ingesting Job Postings...")
        ingest_directory("src/agentic_ai_job_board/rag/data/postings", doc_type="job_posting")

    try:
        client.close()
    except Exception:
        pass
        
    print("Ingestion complete!")