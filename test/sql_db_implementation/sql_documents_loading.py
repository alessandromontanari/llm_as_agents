import unittest
from langchain_community.document_loaders import SQLDatabaseLoader
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine


class TestSQLDocumentsLoading(unittest.TestCase):
    def test_sql_documents_loading(self):

        path_dir = './data/database_sql/test/'
        db_name = 'example.db'

        engine = create_engine(f"sqlite:///{path_dir+db_name}")

        db = SQLDatabase(engine)
        loader = SQLDatabaseLoader(query='SELECT * FROM software', db=db)
        documents = loader.load()

        for doc in documents:
            print("-" * 40)
            print(f"{doc}")
            print("-" * 40)

        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
