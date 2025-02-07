import os
from dotenv import load_dotenv
load_dotenv()

github_token = os.getenv('GITHUB_TOKEN')

import unittest
import sqlite3
import pandas as pd
from urllib.parse import urlparse
import requests

def database_creation():

    test_path = './data/database_sql/test/'

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    database_path = os.path.join(test_path, 'example.db')

    # Connection to database
    conn = sqlite3.connect(database_path)

    # Create a cursor object
    cursor = conn.cursor()

    # Drop the table if it exists
    cursor.execute('DROP TABLE IF EXISTS software')

    # Create the table
    cursor.execute('''
CREATE TABLE software (
    id INTEGER PRIMARY KEY,
    url TEST NOT NULL,
    repo_name TEXT NOT NULL,
    repo_description TEXT NOT NULL,
    stars INTEGER NOT NULL,
    language TEXT NOT NULL
)
''')

    return conn, cursor


class TestSQLDataBaseCreation(unittest.TestCase):
    def test_sql_db_creation(self):

        db_connection, db_cursor = database_creation()

        dataframe = pd.read_csv(
            "./data/dataset/dataset_cross_checked_code_mentions_astroph1_hess0_skiprows0_maxrows337.csv",
            header=0,
            nrows=10,
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
                                language_for_db.append(data['language'] if data['language'] is not None else "Not specified")
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

        # Step 6: Fetch and print the data
        rows = db_cursor.fetchall()
        for row in rows:
            print(row)

        db_connection.close()

        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
