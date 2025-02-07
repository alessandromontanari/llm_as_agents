import unittest
from langchain_community.document_loaders import CSVLoader

class TestDocumentsLoading(unittest.TestCase):
    def test_documents_loading(self):

        loader = CSVLoader(
            file_path="./data/dataset/dataset_cross_checked_code_mentions_astroph1_hess0_skiprows0_maxrows337.csv",
            csv_args={'delimiter': ",", 'fieldnames': ['', 'identifier', 'title', 'keywords', 'category', 'mentioned_software', 'urls']}
        )

        docs = loader.load()

        print(docs[1])

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
