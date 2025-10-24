import pandas as pd
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
# --- TAMBAH IMPORT INI ---
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

DATA_PATH = "Resume.csv"
COLLECTION_NAME = "capstone"
EMBEDDING_DIM = 1536 

def setup_qdrant_client():
    qdrant_client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY,
        timeout=60  
    )
    return qdrant_client

def setup_models():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    return embeddings

def setup_chunker():
    chunker = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', ' '], 
        chunk_size=1000, 
        chunk_overlap=200
    )
    return chunker

def process_and_ingest_data():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File {DATA_PATH} not found.")
        return
    

    df['Resume_str'] = df['Resume_str'].str.strip()

    qdrant_client = setup_qdrant_client()
    embeddings = setup_models()
    chunker = setup_chunker()

    print(f"Recreating collection '{COLLECTION_NAME}'")
    try:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print("Collection recreated.")

        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.category", 
            field_schema=PayloadSchemaType.KEYWORD
        )
        
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.doc_id",
            field_schema=PayloadSchemaType.INTEGER
        )

        print("Payload index created successfully.")
       
    except Exception as e:
        print(f"Error creating collection/index: {e}")
        return


    documents = []
    for index, row in df.iterrows():
        metadata = { 
            'doc_id': row['ID'],
            'category': row['Category']
        }
        document = Document(page_content=row['Resume_str'], metadata=metadata)
        documents.append(document) 

    print(f"Splitting {len(documents)} documents...")
    chunks = chunker.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    if not chunks:
        print("No chunks to ingest.")
        return

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings, 
        vector_name=""
    )

    print(f"Uploading {len(chunks)} chunks in batches...")
    vector_store.add_documents(chunks, batch_size=64) 
    
if __name__ == "__main__":
    process_and_ingest_data()
    print("Ingestion complete.")