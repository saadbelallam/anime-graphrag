import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag123")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.mistral.ai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small-latest")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "384"))

RAW_DIR = "data/raw"


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
