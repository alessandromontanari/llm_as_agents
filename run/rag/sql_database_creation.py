# -*- coding: utf-8 -*-
"""

This script is executable with python -m run.rag.sql_database_creation
"""
import os
from dotenv import load_dotenv

load_dotenv()

os.getenv("MISTRAL_API_KEY")
os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_API_KEY")
os.getenv("HF_TOKEN")
github_token = os.getenv('GITHUB_TOKEN')

from utils.database import database_creation
import pandas as pd
from urllib.parse import urlparse
import requests
from tqdm import tqdm
import logging

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

def loop_for_compiling_sql_database_columns(dataframe, api_base, headers, empty_lists_for_db):

    (identifier_for_software_db, url_for_db, repo_name_for_db, repo_description_for_db, stars_for_db, language_for_db,
     identifier_for_paper_db, title_for_db, keywords_for_db) = empty_lists_for_db

    # TODO: change or remove the length limit for the dataframe. Or make it choosable.

    for ii, row in tqdm(dataframe[0:500].iterrows(), total=len(dataframe[0:500])):

        logging.info(f"Processing {ii+1} over {len(dataframe)} urls entries")

        identifier_for_paper_db.append(row["identifier"])
        title_for_db.append(row["title"])
        keywords_for_db.append(row["keywords"])

        if "git" in row["urls"]:

            # Because in each row of dataframe["urls"], there may be more than one url, so we are checking once at a time
            strip_individual_urls = row["urls"][2:-1].split(" ")

            for individual in strip_individual_urls:

                if "git" in individual:

                    # Extract owner and repo from the URL
                    try:
                        parts = urlparse(individual).path.strip("/").split("/")
                        owner, repo = parts[0], parts[1]

                        # API request to fetch repository details
                        response = requests.get(f"{api_base}/{owner}/{repo}", headers=headers)
                        if response.status_code == 200:
                            # from GitHub API
                            data = response.json()
                            url_for_db.append(individual)
                            repo_name_for_db.append(data['name'])
                            repo_description_for_db.append(data['description'] if data['description'] is not None else "Not specified")
                            stars_for_db.append(data['stargazers_count'])
                            language_for_db.append(data['language'] if data['language'] is not None else "Not specified")

                            identifier_for_software_db.append(row["identifier"])
                        else:
                            # from GitHub API
                            url_for_db.append(individual)
                            repo_name_for_db.append("Failed request")
                            repo_description_for_db.append("Failed request")
                            stars_for_db.append("Failed request")
                            language_for_db.append("Failed request")

                            identifier_for_software_db.append(row["identifier"])

                            logging.info(f"Failed to fetch details for {individual}")
                    except Exception as e:
                        logging.info(f"Skipping because of error {e}")

    for ii, kw in enumerate(keywords_for_db):
        keywords_for_db[ii] = ",".join(kw[1:-1].split(", "))

    return (identifier_for_software_db, url_for_db, repo_name_for_db, repo_description_for_db, stars_for_db, language_for_db,
            identifier_for_paper_db, title_for_db, keywords_for_db)



def main():

    db_papers_connection, db_papers_cursor = database_creation(
        path_dir='./data/database_sql/test/', database_name="paper_info.db", table_name="paper_info"
    )
    db_software_connection, db_software_cursor = database_creation(
        path_dir='./data/database_sql/test/', database_name="software.db", table_name="software"
    )

    dataframe = pd.read_csv(
        "./data/dataset/dataset_cross_checked_code_mentions_astroph1_hess0_skiprows0_maxrows60569.csv",
        header=0,
    )

    # GitHub API base URL
    api_base = "https://api.github.com/repos"

    headers = {"Authorization": f"token {github_token}"}

    identifier_for_software_db = []
    identifier_for_paper_db = []
    title_for_db = []
    keywords_for_db = []
    url_for_db = []
    repo_name_for_db = []
    repo_description_for_db = []
    stars_for_db = []
    language_for_db = []

    empty_lists_for_db = [
        identifier_for_software_db, url_for_db, repo_name_for_db, repo_description_for_db, stars_for_db, language_for_db,
        identifier_for_paper_db, title_for_db, keywords_for_db
    ]

    (identifier_for_software_db, url_for_db, repo_name_for_db, repo_description_for_db, stars_for_db, language_for_db,
     identifier_for_paper_db, title_for_db, keywords_for_db) = loop_for_compiling_sql_database_columns(
        dataframe, api_base, headers, empty_lists_for_db
    )

    # Insert the data in the software database
    data_for_db = list(zip(url_for_db, repo_name_for_db, repo_description_for_db, stars_for_db, language_for_db, identifier_for_software_db))
    db_software_cursor.executemany(f'''
    INSERT INTO software (url, repo_name, repo_description, stars, language, paper_identifier) VALUES (?, ?, ?, ?, ?, ?)
    ''', data_for_db)
    db_software_connection.commit()
    db_software_cursor.execute(f'SELECT * FROM software')

    rows = db_software_cursor.fetchall()
    print("Checking the rows in the database:")
    for row in rows:
        print(row)
    print(len(rows))

    db_software_connection.close()

    # Insert the data in the paper_info database
    data_for_db = list(zip(identifier_for_paper_db, title_for_db, keywords_for_db))
    db_papers_cursor.executemany(f'''
    INSERT INTO paper_info (paper_identifier, paper_title, paper_keywords) VALUES (?, ?, ?)
    ''', data_for_db)
    db_papers_connection.commit()
    db_papers_cursor.execute(f'SELECT * FROM paper_info')

    rows = db_papers_cursor.fetchall()
    print("Checking the rows in the database:")
    for row in rows:
        print(row)
    print(len(rows))

    db_papers_connection.close()

if __name__ == '__main__':
    main()