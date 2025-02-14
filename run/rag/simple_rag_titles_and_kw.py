# -*- coding: utf-8 -*-
"""

This script is executable with python -m run.rag.simple_rag_titles_and_kw
"""
import os
from dotenv import load_dotenv
load_dotenv()

os.getenv("MISTRAL_API_KEY")
os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_API_KEY")
os.getenv("HF_TOKEN")

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain.text_splitter import CharacterTextSplitter
from langchain import hub
from langchain_chroma import Chroma
from langgraph.graph import START, StateGraph
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.dataset import cosine_similarity_search
from utils.rag import State, retrieve_csv_database, generate
from functools import partial
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def setup_initialization():

    llm = ChatMistralAI(model="ministral-3b-latest")
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        wait_time=10,  # Should be used for rate limit retries
        max_retries=5,
        timeout=60
    )

    df = pd.read_csv(
        # "./data/dataset/dataset_cross_checked_code_mentions_astroph1_hess0_skiprows0_maxrows337.csv",
        "./data/dataset/dataset_cross_checked_code_mentions_astroph1_hess0_skiprows0_maxrows60569.csv",
        delimiter=",",
        header=0
    )

    # create documents for titles
    metadata_doc = [{"document": nn} for nn in range(len(df["title"]))]

    splitter = CharacterTextSplitter(
        chunk_size=250,  # max length for each chunk
        chunk_overlap=0,  # up to ten chars overlapping between chunks
        length_function=len,
        is_separator_regex=False,
        separator='\n'
    )

    documents_titles = splitter.create_documents(df["title"], metadatas=metadata_doc)
    documents_keywords = splitter.create_documents(df["keywords"], metadatas=metadata_doc)

    texts_title = [document.page_content for document in documents_titles]
    # texts_keywords = [document.page_content for document in documents_keywords]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts_title)

    query = "gammapy"
    similar_documents_titles, similar_documents_keywords = cosine_similarity_search(
        documents=documents_titles, documents_complement=documents_keywords,
        query=query, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix
    )

    print("Preliminary selection")
    for ii, doc in enumerate(similar_documents_titles):
        print(doc.page_content, similar_documents_keywords[ii].page_content)

    vector_store_titles = Chroma.from_documents(
        documents=similar_documents_titles,
        collection_name="test_titles",
        embedding=embeddings,
        persist_directory=None,
        # Where to save data locally, remove if not necessary
        collection_metadata=None,  # one can add metadatas, Dict
    )
    vector_store_keywords = Chroma.from_documents(
        documents=similar_documents_keywords,
        collection_name="test_keywords",
        embedding=embeddings,
        persist_directory=None,
        # Where to save data locally, remove if not necessary
        collection_metadata=None,  # one can add metadatas, Dict
    )

    return llm, embeddings, df, vector_store_titles, vector_store_keywords


def main():

    llm, embeddings, df, vector_store_titles, vector_store_keywords = setup_initialization()

    print("Setup initialized.")

    prompt = hub.pull("rlm/rag-prompt")

    graph_builder = StateGraph(State)
    partial_retrieve = partial(retrieve_csv_database, vector_store_titles=vector_store_titles, vector_store_keywords=vector_store_keywords)
    partial_generate = partial(generate, prompt=prompt, llm=llm)
    graph_builder.add_sequence([
        ("retrieve", partial_retrieve),
        ("generate", partial_generate)
    ])

    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    print("Graph initialization completed.")

    question = "Can you find anything among the provided titles about gammapy?"

    result = graph.invoke({
        "question": question}
    )

    print("You've asked: \n" + question)

    identified_urls = []

    # TODO: fix to have a better looking output
    # TODO: the generate chat completion finds matching and works only with titles,
    #  but not with the list of keywords.

    print("\nHere are the better matching titles:")
    for context_title in result["context"]:
        row = df.loc[df["title"] == context_title[0].page_content]
        identified_urls.append(row["urls"])

        print(f"{context_title[0].page_content:<{len(context_title[0].page_content) + 4}} score: {context_title[1]}")

    print(f'\nAnswer to your question: {result["answer"]}')

    print(identified_urls)


if __name__ == '__main__':
    main()
