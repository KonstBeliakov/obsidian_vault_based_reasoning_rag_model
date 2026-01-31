import os

from dotenv import load_dotenv
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
from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


load_dotenv(dotenv_path='./data.env') # loading environment variables form data.env

QDRANT_COLLECTION = "districts_rag"
LLM_MODEL = "gpt-4o-mini"
# EMBEDDING_MODEL = "text-embedding-ada-002" #"text-embedding-3-small"


openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

langfuse_public_api_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_private_api_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_base_url = os.getenv("LANGFUSE_BASE_URL")

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0.2,
    api_key=openai_api_key,
)

# Does not work for some reason
# embeddings = OpenAIEmbeddings(
#     model=EMBEDDING_MODEL,
#     api_key=openai_api_key,
# )

#embeddings = HuggingFaceEmbeddings(
#    model_name="all-MiniLM-L6-v2"
#)


embedding_model_name = 'intfloat/multilingual-e5-large'
embedding_model = SentenceTransformer(embedding_model_name)

tavily = TavilyClient(api_key=tavily_api_key)

langfuse = Langfuse(
    public_key=langfuse_public_api_key,
    secret_key=langfuse_private_api_key,
    host=langfuse_base_url,
)


if __name__ == '__main__':
    print('openai, tavily and langfuse apis configured successfully')
    print("Loading embedding model...")
    test_embedding = embedding_model.encode("test")
    print(f"Model loaded successfully! Vector size: {len(test_embedding)}")