import unittest

from tqdm import tqdm
from utils.search_the_web import SearchTheWeb
from bs4 import BeautifulSoup
from types import NoneType

class TestSearchTheWeb(unittest.TestCase):

    # TODO: extend this when more functionalities exist in SearchTheWeb

    def test_search_the_web(self):

        path_to_dataframe = "./data/dataset/dataset_cross_checked_code_mentions_astroph1_hess0_skiprows0_maxrows60569.csv"

        keywords_code = ["code", "pipeline", "software", "development", "programming"]

        search_the_web_for_code = SearchTheWeb(path_to_dataframe=path_to_dataframe, keywords=keywords_code)
        dataframe = search_the_web_for_code.dataframe
        for ii, row in tqdm(dataframe[0:100].iterrows(), total=len(dataframe[0:100])):
            urls = row["urls"].split(' ') if type(row["urls"]) == str else []
            for url in urls:
                if "git" not in url:
                    response: str | None = search_the_web_for_code.fetch_page_content(url)
                    self.assertTrue(type(response) == str or type(response) == NoneType)
                    if response:
                        soup = BeautifulSoup(response, 'html.parser')
                        text_content = soup.get_text()
                        keyword_counts = search_the_web_for_code.search_keywords_in_content(text_content, keywords_code)
                        self.assertTrue(type(keyword_counts) == dict)
                        self.assertGreater(len(keyword_counts), 0)



if __name__ == '__main__':
    unittest.main()
