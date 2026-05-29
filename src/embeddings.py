from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings


from data_parser import parse_files
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

embedding_model = GoogleGenAIEmbedding(
    model_name = "gemini-embedding-2",
    api_key=api_key,
    embed_batch_size = 100,
    )

class Embeddings:
    def __init__(self, file_path):
        self.file_path = file_path

    def get_embeddings(self):
        nodes = parse_files(file_path=self.file_path)
        embeddings = []
        for i in range(len(nodes)):
            node_embedding = embedding_model.get_text_embedding(nodes[i].text)
            embeddings.append(node_embedding)
        return embeddings


