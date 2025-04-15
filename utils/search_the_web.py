import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests

import logging
import os


log_file_path = "./logs/websearch/"

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

class SearchTheWeb:
    def __init__(
            self,
            path_to_data: str,
            keywords: list,
            from_dataframe_or_base_file: str = "dataframe",
    ):

        # from_dataframe_or_base_file either "dataframe" or "base_file" at the moment,
        self.from_dataframe_or_base_file = from_dataframe_or_base_file

        self.path_to_data = path_to_data

        self.identifiers, self.urls = self.preprocess_urls()

        self.keywords = keywords

    @staticmethod
    def fetch_page_content(url, timeout=5):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Raise an error for bad status codes
            logging.info(msg=f"Successful request on {url}")
            return response.text
        except requests.RequestException as e:
            logging.error(msg=f"Error fetching {url}: {e}")
            return None

    @staticmethod
    def search_keywords_in_content(content, keywords):
        content_lower = content.lower()
        found_keywords = {keyword: content_lower.count(keyword) for keyword in keywords}
        return found_keywords

    def preprocess_urls(self):
        # TODO: function to preprocess the urls, because the data structure is different depending on whether the input is a dataframe or the base file
        if self.from_dataframe_or_base_file == "dataframe":
            dataframe = pd.read_csv(self.path_to_data, header=0)
            identifiers = dataframe['identifier']
            urls = dataframe['urls']
        elif self.from_dataframe_or_base_file == "base_file":
            dataframe = pd.read_csv(self.path_to_data, names=["identifier", "urls"], sep="\t")
            identifiers = dataframe['identifier']
            urls = dataframe['urls']
        else:
            raise ValueError("from_dataframe_or_base_file can either be 'dataframe' or 'base_file'")

        return identifiers, urls


    def count_keywords_on_website(self):

        results = {}

        for ii, urls_list in tqdm(enumerate(self.urls), total=len(self.urls)):
            urls_list = urls_list.split(' ') if type(urls_list) == str else []
            for url in urls_list:
                if "git" not in url:
                    logging.info(f"Fetching {url}...")
                    content: str | None = self.fetch_page_content(url)
                    if content:
                        logging.info(f"\tcounting keywords...")
                        soup = BeautifulSoup(content, 'html.parser')
                        text_content = soup.get_text()
                        keyword_counts = self.search_keywords_in_content(text_content, self.keywords)
                        results[url] = keyword_counts

        return results

    def find_link_to_code(self):
        # TODO: this function should find links to code if a code keyword is mentioned.
        #  Like, file with .py, .C, .cpp, etc.
        pass

    # TODO: some function to output more than only the count?
    #  For instance the name of the paper and the mention of the keyword.

    def search(self):

        # if self.from_dataframe_or_base_file == "dataframe":
        # elif
        keywords_count_in_urls = self.count_keywords_on_website()

        return keywords_count_in_urls
