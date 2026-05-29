from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path


def parse_files(file_path):
    reader = FlatReader()
    md_docs = reader.load_data(file_path)

    parser=MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(md_docs)
    return nodes
