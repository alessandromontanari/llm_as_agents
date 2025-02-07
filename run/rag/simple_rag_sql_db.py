# -*- coding: utf-8 -*-
"""

This script is executable with python -m run.rag.simple_rag_sql_db
"""
import os
from dotenv import load_dotenv
load_dotenv()

os.getenv("MISTRAL_API_KEY")
os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_API_KEY")
os.getenv("HF_TOKEN")

import numpy as np
from langchain_community.document_loaders import SQLDatabaseLoader
from langchain_community.utilities.sql_database import SQLDatabase
from langchain import hub
from sqlalchemy import create_engine
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langgraph.graph import START, StateGraph
from langchain_core.vectorstores import InMemoryVectorStore
from functools import partial
from utils.rag import retrieve_sql_database, generate, State
import warnings
warnings.filterwarnings("ignore")

def setup_initialization():

    llm = ChatMistralAI(model="ministral-3b-latest")
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        wait_time=30,  # Should be used for rate limit retries
        max_retries=3,
        timeout=300
    )

    vector_store = InMemoryVectorStore(embeddings)

    path_dir = './data/database_sql/test/'
    db_name = 'example.db'

    engine = create_engine(f"sqlite:///{path_dir + db_name}")

    db = SQLDatabase(engine)
    loader = SQLDatabaseLoader(query='SELECT * FROM software', db=db)
    documents = loader.load()

    documents_for_rag = []

    for doc in documents:
        documents_for_rag.append(
            Document(
                id='', metadata={'document': doc.page_content.split('\n')[0][4:]},
                page_content=doc.page_content.split('repo_description: ')[1].split('\n')[0]
            )
        )

    _ = vector_store.add_documents(documents=documents_for_rag)

    return llm, embeddings, vector_store


def main():

    llm, embeddings, vector_store = setup_initialization()

    prompt = hub.pull("rlm/rag-prompt")

    graph_builder = StateGraph(State)
    partial_retrieve = partial(retrieve_sql_database, vector_store=vector_store)
    partial_generate = partial(generate, prompt=prompt, llm=llm)
    graph_builder.add_sequence([
        ("retrieve", partial_retrieve),
        ("generate", partial_generate)
    ])

    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    question = "Can you find anything among the provided repository descriptions python code for astrophysics?"

    result = graph.invoke({
        "question": question}
    )

    print("You've asked: \n" + question)

    print("\nHere are the better matching descriptions:")
    # max_len = np.max([len(context[0].page_content) for context in result["context"]])

    for context in result["context"]:

        print(f"{context[0].page_content}\n score: {round(context[1], 4)}")

    print(f'\nAnswer to your question: {result["answer"]}')



if __name__ == '__main__':
    main()