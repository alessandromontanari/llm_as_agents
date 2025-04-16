import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests

import logging
import os
import re


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

# TODO: may need to think of more keywords to include
keywords_code = ["code", "codes", "algorithm", "software", "programming", "script", "pipeline", "repository", "notebook", "jupyter", "macro", "development"]

# TODO: may need to remove some extensions, not sure they are all needed...
code_extensions_dict = {
    'python': ['.py', '.pyc', '.pyd', '.pyw', '.pyx'],
    'javascript': ['.js', '.jsx', '.mjs', '.cjs'],
    'java': ['.java', '.class', '.jar'],
    'c': ['.c', '.h'],
    'c++': ['.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.hh'],
    'c#': ['.cs', '.csx'],
    'ruby': ['.rb', '.rbw', '.gemspec'],
    'php': ['.php', '.php3', '.php4', '.php5', '.phtml'],
    'swift': ['.swift'],
    'objective-c': ['.m', '.h'],
    'go': ['.go'],
    'rust': ['.rs'],
    'kotlin': ['.kt', '.kts'],
    'typescript': ['.ts', '.tsx'],
    'perl': ['.pl', '.pm'],
    'shell': ['.sh', '.bash', '.zsh', '.csh', '.ksh'],
    'r': ['.r', '.R'],
    'matlab': ['.m'],
    'html': ['.html', '.htm'],
    'css': ['.css'],
    'sql': ['.sql'],
    'markdown': ['.md', '.markdown'],
    'dart': ['.dart'],
    'scala': ['.scala'],
    'haskell': ['.hs'],
    'lua': ['.lua'],
    'groovy': ['.groovy'],
    'erlang': ['.erl'],
    'elixir': ['.ex', '.exs'],
    'julia': ['.jl'],
    'coffeescript': ['.coffee'],
    'f#': ['.fs', '.fsx'],
    'visual basic': ['.vb', '.vbs'],
    'assembly': ['.asm', '.s'],
    'racket': ['.rkt'],
    'scheme': ['.scm', '.ss'],
    'prolog': ['.pl', '.pro'],
    'ada': ['.adb', '.ads'],
    'fortran': ['.f', '.for', '.f90', '.f95'],
    'cobol': ['.cbl', '.cob', '.cpy'],
    'zip': ['.tar', '.tar.gz', '.gz', '.zip', '.tgz']
}


class SearchTheWeb:
    def __init__(
            self,
            path_to_data: str,
            keywords: [list],
            from_dataframe_or_base_file: str = "dataframe",
            max_urls_search: int | None = 100,
    ):

        # from_dataframe_or_base_file either "dataframe" or "base_file" at the moment,
        self.from_dataframe_or_base_file = from_dataframe_or_base_file
        self.max_urls_search = max_urls_search

        self.path_to_data = path_to_data

        self.identifiers, self.urls = self.preprocess_urls()

        self.keywords = keywords[0]
        self.code_extensions = keywords[1]

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
        found_keywords = {keyword: len(re.findall(r'\b' + re.escape(keyword) + r'\b', content_lower)) for keyword in keywords}
        return found_keywords

    @staticmethod
    def search_code_referenced_in_content(content, code_extensions, keyword):

        # Tokenize the content into words
        words = re.findall(r'\b\w+\b', content.lower())
        found_extensions = {code_extension: 0 for code_extension in code_extensions}

        for i, word in enumerate(words):
            if word in code_extensions:
                start = max(0, i - 10)  # if the keyword is within 10 words before or after the code extension
                end = min(len(words), i + 11)
                if keyword in words[start:end]:
                    found_extensions[word] += 1
        return found_extensions

    def preprocess_urls(self):
        if self.from_dataframe_or_base_file == "dataframe":
            dataframe = pd.read_csv(self.path_to_data, header=0)
            identifiers = dataframe['identifier']
            urls = dataframe['urls']
            logging.info(msg=f"Successfully processed the input dataframe.")
        elif self.from_dataframe_or_base_file == "base_file":
            # now the basefile considered are the ones with removed urls, but merged into complete year ranges (data/database_with_urls/complete/database_urls_1901-1912.csv)
            dataframe = pd.read_csv(self.path_to_data, header=0, index_col=0, sep='\t')
            identifiers = dataframe['identifier']
            urls = dataframe['url']
            logging.info(msg=f"Successfully processed the input file.")
        else:
            raise ValueError("from_dataframe_or_base_file can either be 'dataframe' or 'base_file'")

        return identifiers, urls

    def count_keywords_on_website(self):

        results_code_mentions = {}
        results_code_extensions = {}

        for ii, urls_list in tqdm(
                enumerate(self.urls[0:self.max_urls_search] if self.max_urls_search is not None else self.urls),
                total=len(self.urls[0:self.max_urls_search])  if self.max_urls_search is not None else len(self.urls)
        ):
            urls_list = urls_list.split(' ') if type(urls_list) == str else []
            for url in urls_list:
                if "git" or "zenodo" not in url:
                    logging.info(f"Fetching {url}...")
                    content: str | None = self.fetch_page_content(url)
                    if content:
                        logging.info(f"\tcounting keywords...")
                        soup = BeautifulSoup(content, 'html.parser')
                        text_content = soup.get_text()
                        keyword_counts = self.search_keywords_in_content(text_content, self.keywords)
                        results_code_mentions[url] = keyword_counts
                        most_cited_keyword = list(keyword_counts.items())[np.argmax([key_count[1] for key_count in list(keyword_counts.items())])][0]
                        code_extensions_counts = self.search_code_referenced_in_content(text_content, self.code_extensions, most_cited_keyword)
                        results_code_extensions[url] = code_extensions_counts

        return results_code_mentions, results_code_extensions

    def find_link_to_code(self):
        # TODO: this function should find links to code if a code keyword is mentioned.
        #  Like, file with .py, .C, .cpp, etc.
        pass

    # TODO: some function to output more than only the count?
    #  For instance the name of the paper and the mention of the keyword.

    def search(self):

        keywords_count_in_urls, code_extension_count_in_urls = self.count_keywords_on_website()

        return keywords_count_in_urls, code_extension_count_in_urls
