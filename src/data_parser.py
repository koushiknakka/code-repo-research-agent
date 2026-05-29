from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path

reader = FlatReader()
md_docs = reader.load_data(Path('data/sample.md'))

parser=MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(md_docs)
