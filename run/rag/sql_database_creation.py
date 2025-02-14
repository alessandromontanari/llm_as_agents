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

def main():

    db_connection, db_cursor = database_creation(path_dir='./data/database_sql/test/', database_name="example.db")

    dataframe = pd.read_csv(
        "./data/dataset/dataset_cross_checked_code_mentions_astroph1_hess0_skiprows0_maxrows60569.csv",
        header=0,
    )

    # GitHub API base URL
    api_base = "https://api.github.com/repos"

    headers = {"Authorization": f"token {github_token}"}

    url_for_db = []
    repo_name_for_db = []
    repo_description_for_db = []
    stars_for_db = []
    language_for_db = []

    # Fetch repository details
    for url in dataframe["urls"]:

        if "git" in url:

            strip_individual_urls = url[2:-1].split(" ")

            for individual in strip_individual_urls:

                if "git" in individual:

                    # Extract owner and repo from the URL
                    try:
                        parts = urlparse(individual).path.strip("/").split("/")
                        owner, repo = parts[0], parts[1]

                        # API request to fetch repository details
                        response = requests.get(f"{api_base}/{owner}/{repo}", headers=headers)
                        if response.status_code == 200:
                            data = response.json()
                            url_for_db.append(individual)
                            repo_name_for_db.append(data['name'])
                            repo_description_for_db.append(data['description'])
                            stars_for_db.append(data['stargazers_count'])
                            language_for_db.append(
                                data['language'] if data['language'] is not None else "Not specified")
                        else:
                            print(f"Failed to fetch details for {individual}")
                    except Exception as e:
                        print(f"Skipping because of error {e}")

    # Insert the data in the database
    data_for_db = list(zip(url_for_db, repo_name_for_db, repo_description_for_db, stars_for_db, language_for_db))
    db_cursor.executemany('''
    INSERT INTO software (url, repo_name, repo_description, stars, language) VALUES (?, ?, ?, ?, ?)
    ''', data_for_db)

    db_connection.commit()

    db_cursor.execute('SELECT * FROM software')

    rows = db_cursor.fetchall()
    for row in rows:
        print(row)

    db_connection.close()


if __name__ == '__main__':
    main()