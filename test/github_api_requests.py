import os
from dotenv import load_dotenv
load_dotenv()

github_token = os.getenv('GITHUB_TOKEN')

import unittest

import pandas as pd
from urllib.parse import urlparse
import requests


class TestGitHubApiRequest(unittest.TestCase):
    def test_github_api_request(self):

        dataframe = pd.read_csv(
            "./data/dataset/dataset_cross_checked_code_mentions_astroph1_hess0_skiprows0_maxrows337.csv",
            header=0,
            nrows=10,
        )

        # GitHub API base URL
        api_base = "https://api.github.com/repos"

        headers = {"Authorization": f"token {github_token}"}

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
                                print(f"Repository: {data['name']}")
                                print(f"Description: {data['description']}")
                                print(f"Stars: {data['stargazers_count']}")
                                print(f"Language: {data['language']}")
                                print("-" * 40)
                            else:
                                print(f"Failed to fetch details for {individual}")
                                print("-" * 40)
                        except Exception as e:
                            print(f"Skipping because of error {e}")
                            print("-" * 40)


if __name__ == '__main__':
    unittest.main()
