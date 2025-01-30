import unittest
import yake
import pandas as pd

class TestKeywordExtractor(unittest.TestCase):

    def test_keyword_extractor(self):

        dataframe = pd.read_csv(
            "./data/database/abstracts.csv",
            header=0,
            sep='\t',
            nrows=10,
            on_bad_lines='skip',
        )

        language = "en"
        max_ngram_size = 3

        kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, top=6)

        for index, content in enumerate(dataframe["description"]):

            abstract = content.split(";Comment")[0]

            keywords = kw_extractor.extract_keywords(abstract)
            keywords = [kw[0] for kw in keywords]

            print(index, f"Title: {dataframe.iloc[index]["title"]}, Keywords: {keywords}")


if __name__ == '__main__':
    unittest.main()
