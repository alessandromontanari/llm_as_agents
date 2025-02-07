import os
from dotenv import load_dotenv
load_dotenv()

os.getenv("MISTRAL_API_KEY")
os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_API_KEY")
os.getenv("HF_TOKEN")

import unittest
from langchain_community.document_loaders import SQLDatabaseLoader
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# TODO: MultiVectorRetriever may be the way to go to get more page_content for a single id


class TestFromSQLDBtoVectorStore(unittest.TestCase):
    def test_from_sqldb_to_vector_store(self):

        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            wait_time=10,  # Should be used for rate limit retries
            max_retries=5,
            timeout=60
        )

        # It persists in memory only during the execution of the script. It is eliminated afterwards.
        vector_store = InMemoryVectorStore(embeddings)

        path_dir = './data/database_sql/test/'
        db_name = 'example.db'

        engine = create_engine(f"sqlite:///{path_dir+db_name}")

        db = SQLDatabase(engine)
        loader = SQLDatabaseLoader(query='SELECT * FROM software', db=db)
        documents = loader.load()

        documents_for_rag = []

        for doc in documents:
            documents_for_rag.append(
                Document(
                    id='', metadata={'document': doc.page_content.split('\n')[0][4:]},
                    page_content=doc.page_content.split('repo_description: ')[1].split('\n')[0]
                )
            )

        vs = vector_store.add_documents(documents=documents_for_rag)

        print(vs)

        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()