from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from qdrant_client import QdrantClient


from data_parser import parse_files
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

client = QdrantClient(location=":memory:")

embedding_model = GoogleGenAIEmbedding(
    model_name = "gemini-embedding-2",
    api_key=api_key,
    embed_batch_size = 100,
    )

Settings.embed_model = embedding_model

vector_store = QdrantVectorStore(
    collection_name="codebase_embeddings",
    client=client,
    prefer_grpc=True,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

class get_embeddings:
    def __init__(self, file_path):
        self.file_path = file_path

    def get_embeddings(self):
        nodes = parse_files(file_path=self.file_path)

        index = VectorStoreIndex(nodes, storage_context=storage_context)


if __name__ == "__main__":
    embedder = get_embeddings(file_path=Path("data/sample.md"))
    embedder.get_embeddings()
    print("🚀 Optimization Check: Vector Index successfully computed and stored in Qdrant!")

