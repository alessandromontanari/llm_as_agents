import unittest

from tqdm import tqdm
from utils.search_the_web import SearchTheWeb, keywords_code, code_extensions_dict
from bs4 import BeautifulSoup
from types import NoneType
import numpy as np

class TestSearchTheWeb(unittest.TestCase):

    # TODO: extend this when more functionalities exist in SearchTheWeb

    def test_search_the_web(self):

        path_to_dataframe = "./data/dataset/dataset_cross_checked_code_mentions_astroph1_hess0_skiprows0_maxrows60569.csv"

        search_the_web_for_code = SearchTheWeb(path_to_data=path_to_dataframe, keywords=keywords_code, from_dataframe_or_base_file="dataframe")
        urls = search_the_web_for_code.urls
        for ii, urls_list in tqdm(enumerate(urls[0:100]), total=len(urls[0:100])):
            urls_list = urls_list.split(' ') if type(urls_list) == str else []
            for url in urls_list:
                if "git" not in url:
                    response: str | None = search_the_web_for_code.fetch_page_content(url)
                    self.assertTrue(type(response) == str or type(response) == NoneType)
                    if response:
                        soup = BeautifulSoup(response, 'html.parser')
                        text_content = soup.get_text()
                        keyword_counts = search_the_web_for_code.search_keywords_in_content(text_content, keywords_code)
                        self.assertTrue(type(keyword_counts) == dict)
                        self.assertGreater(len(keyword_counts), 0)
                        most_cited_keyword = list(keyword_counts.items())[np.argmax([key_count[1] for key_count in list(keyword_counts.items())])][0]
                        code_extensions_counts = search_the_web_for_code.search_code_referenced_in_content(text_content, code_extensions_dict, most_cited_keyword)
                        self.assertTrue(type(code_extensions_counts) == dict)
                        self.assertGreater(len(code_extensions_counts), 0)


if __name__ == '__main__':
    unittest.main()
