from dotenv import load_dotenv
import openai
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

documents = []
directory_path = 'documents'
identifier=0
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    with open(file_path, 'r') as file:
        content = file.read()
    identifier += 1
    document = Document(
        page_content=content,
        metadata={"title": filename},
        id=identifier
    )
    documents.append(document)

uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)

results = vector_store.similarity_search(
    "Ist es sinnvoll, zu studieren`?",
    k=2,
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")