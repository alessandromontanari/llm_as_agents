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

from langchain_community.document_loaders import SQLDatabaseLoader
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from langchain_mistralai import ChatMistralAI
from utils.rag import similarity_search, find_semantically_similar_text

import warnings
import logging
import yake
import re

warnings.filterwarnings("ignore")

log_file_path = "./logs/rag/"

if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)

full_path = __file__

script_name = os.path.basename(full_path)[:-3]  # remove .py from the end

logging.basicConfig(
    filename=log_file_path+script_name+".log",
    format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s: %(message)s",
    level=logging.INFO,
    filemode='w'
)


def search_from_query(query):

    llm = ChatMistralAI(model="ministral-3b-latest")
    # TODO: am I sure I don't need to use embeddings? Would this improve at all the current version of the code?
    # embeddings = MistralAIEmbeddings(
    #     model="mistral-embed",
    #     wait_time=30,  # Should be used for rate limit retries
    #     max_retries=3,
    #     timeout=300
    # )

    path_dir = './data/database_sql/test/'
    db_software_name, db_paper_info_name = 'software.db', 'paper_info.db'

    engine_software_db = create_engine(f"sqlite:///{path_dir + db_software_name}")

    db = SQLDatabase(engine_software_db)
    loader_software_db = SQLDatabaseLoader(query='SELECT * FROM software', db=db)
    documents_software_db = loader_software_db.load()

    engine_paper_info_db = create_engine(f"sqlite:///{path_dir + db_paper_info_name}")

    db = SQLDatabase(engine_paper_info_db)
    loader_paper_info_db = SQLDatabaseLoader(query='SELECT * FROM paper_info', db=db)
    documents_paper_info_db = loader_paper_info_db.load()

    all_urls = {}
    for doc in documents_software_db:
        all_urls[int(doc.page_content.split('paper_identifier: ')[1])] = doc.page_content.split('url: ')[1].split('\n')[0]

    sim_search = bool(
        input("Do you wish to also scan the database for text semantically similar to the query? (Press any key and ENTER for True or just ENTER for False.) ")
    )
    if sim_search:
        print("Generating semantically similar text.")
        synonyms = find_semantically_similar_text(query, llm)
    else:
        print("Using only the query.")
        synonyms = query

    queries = re.split(",|, |\n|,\n", synonyms)

    cleaned_queries = []
    for ii, string in enumerate(queries):
        cleaned_string = re.sub(r'[^a-zA-Z\s]', ' ', string)
        if ii == 0:
            cleaned_queries.append(query)
        cleaned_queries.append(cleaned_string.strip())

    similar_repo_name_documents, repo_names_sim_score, urls_from_repo_names = similarity_search(
        documents=documents_software_db, field_name="repo_name", query=cleaned_queries
    )
    similar_repo_description_documents, repo_descr_sim_score, urls_from_repo_descr = similarity_search(
        documents=documents_software_db, field_name="repo_description", query=cleaned_queries
    )
    similar_title_documents, titles_sim_score = similarity_search(
        documents=documents_paper_info_db, field_name="paper_title", query=cleaned_queries
    )
    similar_keyword_documents, keywords_sim_score = similarity_search(
        documents=documents_paper_info_db, field_name="paper_keywords", query=cleaned_queries
    )

    titles = {}
    for doc in similar_title_documents:
        titles[int(doc.metadata["document"])] = doc.page_content

    ids_titles = set([int(doc.metadata["document"]) for doc in similar_title_documents])
    ids_kws = set([int(doc.metadata["document"]) for doc in similar_keyword_documents])
    ids_repo_descr = set([doc.metadata["document"] for doc in similar_repo_description_documents])

    intersection_titles_kws = list(ids_titles & ids_kws)
    intersection_titles_repo_descr = list(ids_titles & ids_repo_descr)

    # TODO: may be useful to create a hierarchy for the responses depending on the stars one repository has
    # TODO: add a way to specify the language

    print("-" * 40)
    print("From the repository descriptions, these are the first five, by score:")
    for ii, doc in enumerate(similar_repo_description_documents[:10]):
        print(f"Repo description: {doc.page_content},"
              f"\n  score: {round(titles_sim_score[ii], 3):3}/1,\n  url: {urls_from_repo_descr[ii]}")
        print("-" * 15)
    if intersection_titles_kws:
        print("-" * 40)
        print("From results from both the paper titles and keywords:")
        for index in intersection_titles_kws:
            try:
                print(all_urls[index])
            except Exception as e:
                print(f"  No url was included with the paper title '{titles[index]}'")
    if intersection_titles_repo_descr:
        print("-" * 40)
        print("From results from both the paper titles and repository descriptions:")
        for index in intersection_titles_repo_descr:
            try:
                print(all_urls[index])
            except Exception as e:
                print("  No url was included with the paper title")

    # print("Most similar titles:")
    # for ii, doc in enumerate(similar_title_documents):
    #     print(f"ID: {doc.metadata["document"]}, {doc.page_content}, \n  score: {round(titles_sim_score[ii], 3):3}/1")
    # print("-" * 40)
    # print("Most similar keywords:")
    # for ii, doc in enumerate(similar_keyword_documents):
    #     print(f"ID: {doc.metadata["document"]}, {doc.page_content}, \n  score: {round(keywords_sim_score[ii], 3):3}/1")
    # print("-" * 40)
    # print("Most similar repository names:")
    # for ii, doc in enumerate(similar_repo_name_documents):
    #     print(f"ID: {doc.metadata["document"]}, {doc.page_content}, \n  score: {round(repo_names_sim_score[ii], 3):3}/1")
    # print("-" * 40)
    # print("Most similar repository descriptions:")
    # for ii, doc in enumerate(similar_repo_description_documents):
    #     print(f"ID: {doc.metadata["document"]}, {doc.page_content}, \n  score: {round(repo_descr_sim_score[ii], 3):3}/1")
    # print("-" * 40)


def main():

    query = "gamma-ray astronomy."

    print(f"You've asked for: \n{query}")

    search_from_query(query)


if __name__ == '__main__':
    main()