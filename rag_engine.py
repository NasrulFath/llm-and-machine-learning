from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import pandas as pd
import os

def build_index(file_obj, storage_dir: str = "vector_store"):
    df = pd.read_csv(file_obj)
    content = df.to_markdown(index=False)
    docs = [Document(text=content)]

    embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    index.storage_context.persist(storage_dir)
    return index

def load_index(storage_dir: str = "vector_store"):
    embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    return load_index_from_storage(storage_context, embed_model=embed_model)

def ask_rag(query: str):
    if not os.path.exists("vector_store/docstore.json"):
        return "❌ Index belum dibuat. Klik tombol 'Bangun Index' dulu."

    llm = OpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    index = load_index()
    chat_engine = index.as_chat_engine(llm=llm, verbose=True)
    return chat_engine.chat(query).response
