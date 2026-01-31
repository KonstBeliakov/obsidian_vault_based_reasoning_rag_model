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


load_dotenv(dotenv_path='./data.env') # loading environment variables form data.env

QDRANT_COLLECTION = "districts_rag"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


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

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=openai_api_key,
)

tavily = TavilyClient(api_key=tavily_api_key)

langfuse = Langfuse(
    public_key=langfuse_public_api_key,
    secret_key=langfuse_private_api_key,
    host=langfuse_base_url,
)


if __name__ == '__main__':
    print('openai, tavily and langfuse apis configured successfully')
