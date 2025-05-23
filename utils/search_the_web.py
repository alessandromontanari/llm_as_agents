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
keywords_code = ["code", "codes", "algorithm", "algorithms", "software",
                 "programming", "script", "pipeline", "repository", "notebook", "jupyter", "macro", "development"]

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

# TODO: SearchTheWeb may become the basis class with fetch_page_content as method,
#  from which two other classes, one searching for keywords and another one searching for the title/h1/meta, inherit

class SearchTheWeb:
    def __init__(
            self,
            path_to_data: str,
            keywords: [list] = None,
            from_dataframe_or_base_file: str = "dataframe",
            max_url_lists_search: int | None = 100,
            shuffle: bool = False
    ):

        # from_dataframe_or_base_file either "dataframe" or "base_file" at the moment,
        self.from_dataframe_or_base_file = from_dataframe_or_base_file
        self.max_url_lists_search = max_url_lists_search
        self.shuffle = shuffle

        self.path_to_data = path_to_data

        self.identifiers, self.urls = self.preprocess_urls()

        if keywords is not None:
            self.keywords = keywords[0]
            self.code_extensions = keywords[1]
        else:
            self.keywords = None
            self.code_extensions = None

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

    # the first inheriting class could include the methods from the next one

    @staticmethod
    def search_keywords_in_content(content, keywords):
        content_lower = content.lower()
        found_keywords = {keyword: len(re.findall(r'\b' + re.escape(keyword) + r'\b', content_lower)) for keyword in keywords}
        return found_keywords

    # TODO: complete logging for the subsequent functions?

    @staticmethod
    def search_code_referenced_in_content(content, code_extensions, keyword):

        # Tokenize the content into words
        words = re.findall(r'\b\w+\b', content.lower())
        found_extensions = {code_extension: 0 for code_extension in code_extensions}

        for i, word in enumerate(words):
            if word in code_extensions:
                # start = max(0, i - 10)  # if the keyword is within 10 words before or after the code extension
                # end = min(len(words), i + 11)
                if keyword in words:
                # if keyword in words[start:end]:
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

    def count_keywords_on_website_or_git_in_url(self):

        results_keyword_mentions = {}
        results_code_extensions = {}
        results_git_in_urls = 0

        if self.shuffle:
            np.random.seed(42)
            urls = pd.Series(np.random.permutation(self.urls))
        else:
            urls = self.urls

        for ii, urls_list in tqdm(
                enumerate(urls[0:self.max_url_lists_search] if self.max_url_lists_search is not None else urls),
                total=len(urls[0:self.max_url_lists_search]) if self.max_url_lists_search is not None else len(urls),
                desc="Counting kws and code exts..."
        ):
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
                        results_keyword_mentions[url] = keyword_counts
                        most_cited_keyword = list(keyword_counts.items())[np.argmax([key_count[1] for key_count in list(keyword_counts.items())])][0]
                        code_extensions_counts = self.search_code_referenced_in_content(text_content, self.code_extensions, most_cited_keyword)
                        results_code_extensions[url] = code_extensions_counts
                else:
                    results_git_in_urls += 1

        return results_keyword_mentions, results_code_extensions, results_git_in_urls

    def count_per_matching_urls(self, keywords_count):

        total_keywords_count_per_url = {}
        for url, kw_counts in keywords_count.items():
            total_keywords_count_per_url[url] = 0
            for _, count in kw_counts.items():
                total_keywords_count_per_url[url] += count

        return total_keywords_count_per_url

    def count_per_keyword(self, keywords_count):

        total_count_per_kw = {}
        for kw in self.keywords:
            total_count_per_kw[kw] = 0
        for url, kw_counts in keywords_count.items():
            for kw, count in kw_counts.items():
                total_count_per_kw[kw] += count

        return total_count_per_kw

    def extensions_count_per_matching_url(self, code_extensions_count):

        total_code_exts_count_per_url = {}
        for url, code_ext_counts in code_extensions_count.items():
            total_code_exts_count_per_url[url] = 0
            for _, count in code_ext_counts.items():
                total_code_exts_count_per_url[url] += count

        return total_code_exts_count_per_url

    def count_positive_urls(self, keywords_count):

        positive_urls_count = 0
        list_positive_urls = []
        for url, kw_counts in keywords_count.items():
            # non_zero_counts = np.count_nonzero([count for _, count in kw_counts.items()])
            non_zero_counts = np.array([count for _, count in kw_counts.items()])
            if non_zero_counts[non_zero_counts > 0].sum() >= 1:  # summing the number of keyword hits to get the total
                positive_urls_count += 1
                list_positive_urls.append(url)

        return positive_urls_count, list_positive_urls

    def count_true_positive_urls(self, keywords_count, code_extensions_count):
        # Assuming a true positive means that in the website there is either two mentions of keywords or a mention of a keyword and a code extension

        true_positive_urls_count = 0
        list_true_positive_urls = []
        for url, kw_counts in keywords_count.items():
            # non_zero_counts = np.count_nonzero([count for _, count in kw_counts.items()])
            # non_zero_ext_counts = np.count_nonzero([count for _, count in code_extensions_count[url].items()])
            non_zero_counts = np.array([count for _, count in kw_counts.items()])
            non_zero_ext_counts = np.array([count for _, count in code_extensions_count[url].items()])
            if non_zero_counts[non_zero_counts > 0].sum() >= 3:  # summing the number of keyword hits to get the total
                true_positive_urls_count += 1
                list_true_positive_urls.append(url)
            elif 1 <= non_zero_counts[non_zero_counts > 0].sum() < 3 and non_zero_ext_counts[non_zero_ext_counts > 0].sum() >=1:  # summing the number of code extension hits
                true_positive_urls_count += 1
                list_true_positive_urls.append(url)

        return true_positive_urls_count, list_true_positive_urls

    # TODO: some more functions?
    #  For instance the name of the paper and the mention of the keyword.

    def search(self):

        assert self.keywords is not None and self.code_extensions is not None, "Cannot process the search if the keywords and code extensions have not been initialized"

        keywords_count_in_urls, code_extension_count_in_urls, git_in_urls_count = self.count_keywords_on_website_or_git_in_url()

        positive_urls_count, positive_urls_list = self.count_positive_urls(keywords_count=keywords_count_in_urls)
        true_positive_urls_count, true_positive_urls_list = self.count_true_positive_urls(
            keywords_count=keywords_count_in_urls, code_extensions_count=code_extension_count_in_urls
        )
        # keywords_count = self.count_per_matching_urls(keywords_count=keywords_count_in_urls)
        keywords_count = self.count_per_keyword(keywords_count=keywords_count_in_urls)

        counts = [git_in_urls_count, positive_urls_count, true_positive_urls_count]
        lists = [positive_urls_list, true_positive_urls_list]

        return keywords_count_in_urls, code_extension_count_in_urls, counts, lists, keywords_count

    # the second inheriting class could include the methods from the next one

    @staticmethod
    def get_title(page_html):
        soup = BeautifulSoup(page_html, 'html.parser')
        title_tag = soup.find('title')  # looking for the <title> tag in the <head> section
        if title_tag:
            return title_tag.get_text(strip=True)
        return None

    @staticmethod
    def get_meta_description(page_html):
        soup = BeautifulSoup(page_html, 'html.parser')
        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description and meta_description.has_attr('content'):  # looking for the content in the <meta> tag with name 'description'
            return meta_description['content']
        return None

    @staticmethod
    def get_og_title(page_html):
        soup = BeautifulSoup(page_html, 'html.parser')
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.has_attr('content'):  # looking for the content in the <meta> tag with property 'og:title'
            return og_title['content']
        return None

    @staticmethod
    def get_first_h1(page_html):
        soup = BeautifulSoup(page_html, 'html.parser')
        h1_tag = soup.find('h1')  # looking for the first <h1> in the <body> section
        if h1_tag:
            return h1_tag.get_text(strip=True)
        return None

    @staticmethod
    def get_first_b(page_html):
        soup = BeautifulSoup(page_html, 'html.parser')
        b_tag = soup.find('b')  # looking for the first <> in the <body> section
        if b_tag:
            return b_tag.get_text(strip=True)
        return None

    def get_page_title(self, page_html):

        title = self.get_title(page_html)
        og_title = self.get_og_title(page_html)
        meta_description = self.get_meta_description(page_html)
        first_h1 = self.get_first_h1(page_html)
        first_b = self.get_first_b(page_html)

        results = [title, meta_description, og_title, first_h1, first_b]
        filtered_results = []

        for result in results:
            if result and not len(result.split()) == 1:
                filtered_results.append(result)

        if title and meta_description and title in meta_description.split():
            filtered_results = [res for res in filtered_results if res != title]

        if filtered_results:
            return ' - '.join(filtered_results), {
            'title': title,
            'og_title': og_title,
            'meta_description': meta_description,
            'first_h1': first_h1,
            'first_b': first_b,
        }
        else:
            return {
            'title': title,
            'og_title': og_title,
            'meta_description': meta_description,
            'first_h1': first_h1,
            'first_b': first_b,
        }