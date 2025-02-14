# -*- coding: utf-8 -*-
"""

This script is executable with python -m run.rag.write_chroma_db
"""
import os
from dotenv import load_dotenv
load_dotenv()

os.getenv("MISTRAL_API_KEY")
os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_API_KEY")
os.getenv("HF_TOKEN")

from langchain_mistralai import MistralAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
import pandas as pd
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")


def add_documents_with_progress(documents, embedding_function, collection_name: str, batch_size: int = 10):
    chroma_db = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory="./data/vector_stores/chroma_langchain_db_"+collection_name,
        collection_metadata=None
    )

    for ii in tqdm(range(0, len(documents), batch_size), desc="Adding Documents to Chroma"):
        batch_documents = documents[ii:ii + batch_size]
        chroma_db.add_documents(batch_documents)
        time.sleep(0.00001)

    chroma_db.persist()


def initialize_and_write_chroma_db():

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        wait_time=10,  # Should be used for rate limit retries
        max_retries=5,
        timeout=60
    )

    df = pd.read_csv(
        "./data/dataset/dataset_cross_checked_code_mentions_astroph1_hess0_skiprows0_maxrows60569.csv",
        delimiter=",",
        header=0
    )

    # create documents for titles
    metadata_doc = [{"document": nn} for nn in range(len(df["title"]))]

    splitter = CharacterTextSplitter(
        chunk_size=100,  # max length for each chunk
        chunk_overlap=0,  # up to ten chars overlapping between chunks
        length_function=len,
        is_separator_regex=False,
        separator='\n'
    )

    documents_title = splitter.create_documents(df["title"], metadatas=metadata_doc)
    documents_keywords = splitter.create_documents(df["keywords"], metadatas=metadata_doc)

    add_documents_with_progress(
        documents=documents_title,
        collection_name="test_titles",
        embedding_function=embeddings,
    )

    add_documents_with_progress(
        documents=documents_keywords,
        collection_name="test_keywords",
        embedding_function=embeddings,
    )


def main():

    initialize_and_write_chroma_db()



if __name__ == '__main__':
    main()
