import json
import uuid
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse import Langfuse, observe
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tavily import TavilyClient

from ai_utils import *
from prompts import react_system_prompt, answer_node_system_prompt, gather_node_system_prompt, \
    analyze_node_system_prompt
from travel_state import TravelState
from utils import FILE_SEP_TOKEN

from sentence_transformers import SentenceTransformer

QDRANT_PATH = "./qdrant_storage"
COLLECTION_NAME = "obsidian_vault_collection"
SEP_TOKENS = [FILE_SEP_TOKEN, '\n---\n', '\n# ', '\n## ', '\n#### ', '\n##### ', '\n###### ', '\n\n', '\n']



@tool
def web_search(query: str) -> Dict[str, Any]:
    """
    Поиск через Tavily.
    Обязательно указывать:
    - query: конкретный вопрос (например, "visa requirements for Russian citizens to Spain in 2026")
    Возвращает:
    - "results": список {"title", "url", "content"}
    """
    res = tavily.search(query=query, max_results=5, country='russia', search_depth="advanced")
    return res


tools = [web_search]
qdrant = None

def check_collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if a collection exists in Qdrant."""
    try:
        collections = client.get_collections().collections
        return any(collection.name == collection_name for collection in collections)
    except Exception as e:
        print(f"Error checking collection: {e}")
        return False


def get_collection_info(client: QdrantClient, collection_name: str):
    """Get information about the collection."""
    try:
        info = client.get_collection(collection_name)
        return info
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return None


def create_qdrant_database_from_obsidian_vault(
        folder='./Main vault',
        force_recreate=False
):
    """
    Create or load a Qdrant vector database from an Obsidian vault.

    Args:
        folder: Path to the Obsidian vault
        force_recreate: If True, recreate the database even if it exists
    """
    global qdrant

    if qdrant is None:
        print(f"Connecting to Qdrant database at {QDRANT_PATH}...")
        qdrant = QdrantClient(path=QDRANT_PATH)

    # Check if collection already exists
    collection_exists = check_collection_exists(qdrant, COLLECTION_NAME)

    if collection_exists and not force_recreate:
        print(f"✓ Collection '{COLLECTION_NAME}' already exists!")
        info = get_collection_info(qdrant, COLLECTION_NAME)
        if info:
            print(f"  - Vectors count: {info.points_count}")
            print(f"  - Vector size: {info.config.params.vectors.size}")
        print("Loading existing database...")
        return qdrant

    if collection_exists and force_recreate:
        print(f"⚠ Collection '{COLLECTION_NAME}' exists but force_recreate=True")
        print("Deleting existing collection...")
        qdrant.delete_collection(COLLECTION_NAME)

    # Create new collection
    print(f"Creating new collection '{COLLECTION_NAME}'...")

    # Read and process the data
    with open('all_data.txt', 'r', encoding='utf-8') as f:
        rules = f.read()[:10_000]
    print(f"Read {len(rules)} characters from all_data.txt")
    print("First 500 characters:")
    print(rules[:500])
    print()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=SEP_TOKENS
    )
    rules_chunks = text_splitter.split_text(rules)
    print(f"Split into {len(rules_chunks)} chunks")
    print("First 3 chunks:")
    for i, chunk in enumerate(rules_chunks[:3]):
        print(f"Chunk {i}: {chunk[:100]}...")
    print()

    vector_size = len(embedding_model.encode("query: test"))
    print(f"Vector size: {vector_size}")

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"✓ Collection created")

    # Create embeddings and insert into database
    print(f"Creating embeddings for {len(rules_chunks)} chunks...")
    points = []
    for idx, doc in enumerate(rules_chunks):
        if idx % 10 == 0:  # Progress indicator
            print(f"  Processing chunk {idx + 1}/{len(rules_chunks)}")

        vec = embedding_model.encode('query: '+ doc)
        points.append(
            PointStruct(
                id=idx,
                vector=vec,
                payload={"text": doc},
            )
        )

    # Insert all points at once
    print("Inserting vectors into database...")
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✓ Inserted {len(points)} vectors")

    # Verify
    info = get_collection_info(qdrant, COLLECTION_NAME)
    if info:
        print(f"\n✓ Database created successfully!")
        print(f"  - Collection: {COLLECTION_NAME}")
        print(f"  - Vectors: {info.points_count}")
        print(f"  - Storage: {QDRANT_PATH}")

    return qdrant


def search_database(query: str, top_k: int = 5):
    """
    Search the vector database.

    Args:
        query: Search query
        top_k: Number of results to return
    """
    print(f"Searching for: '{query}'")

    global qdrant
    if qdrant is None:
        qdrant = QdrantClient(path=QDRANT_PATH)

    # Check if collection exists
    if not check_collection_exists(qdrant, COLLECTION_NAME):
        print(f"Error: Collection '{COLLECTION_NAME}' does not exist!")
        print("Please run create_qdrant_database_from_obsidian_vault() first")
        return []

    # Embed query
    query_vector = embedding_model.encode(query)

    # Search
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )

    points = results.points

    print(f"\nFound {len(points)} results:")
    for i, result in enumerate(points):
        print(f"\n--- Result {i + 1} (score: {result.score:.4f}) ---")
        print(result.payload['text'][:200] + "...")

    return points


def rag_retrieve(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if qdrant is None:
        raise Exception('QdrantClient is not defined')
    vec = embedding_model.encode(query)
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=top_k
    )
    docs = []
    for h in hits.points:
        payload = h.payload
        docs.append(
            {
                "text": payload["text"],
                "category": payload.get("category"),
                "score": h.score,
            }
        )
    return docs


if __name__ == '__main__':
    try:
        create_qdrant_database_from_obsidian_vault(
            folder='./Main vault',
            force_recreate=False
        )

        print("\n" + "=" * 50)
        print("SEARCH EXAMPLE")
        print("=" * 50)

        t = search_database("query: Что ты знаешь про Мику?", top_k=3)
        print(t)

    finally:
        if qdrant is not None:
            print("Closing Qdrant connection...")
            qdrant.close()
