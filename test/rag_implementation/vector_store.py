import os
from dotenv import load_dotenv
load_dotenv()

os.getenv("MISTRAL_API_KEY")
os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_API_KEY")


from langchain_mistralai import MistralAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from langchain_chroma import Chroma
import unittest
import pandas as pd


class TestVectorStore(unittest.TestCase):
    def test_vector_store(self):

        # llm = ChatMistralAI(model="mistral-large-latest")
        embeddings = MistralAIEmbeddings(model="mistral-embed")

        df = pd.read_csv(
            "./data/dataset/dataset_cross_checked_code_mentions_astroph1_hess0_skiprows0_maxrows337.csv",
            delimiter=",",
            header=0
        )

        titles = df["title"][:100]

        # create documents for titles
        metadata_doc = [{"document": nn + 1} for nn in range(len(titles))]

        splitter = CharacterTextSplitter(
            chunk_size=250,  # max length for each chunk
            chunk_overlap=0,  # up to ten chars overlapping between chunks
            length_function=len,
            is_separator_regex=False,
            separator='\n'
        )

        documents_title = splitter.create_documents(titles, metadatas=metadata_doc)  # to isolate each title

        vector_store = Chroma.from_documents(
            documents=documents_title,
            collection_name="test",
            embedding=embeddings,
            persist_directory="./data/vector_stores/chroma_langchain_db_test",  # Where to save data locally, remove if not necessary
            collection_metadata=None,  # one can add metadatas, Dict
        )

        collection_metadatas = vector_store._collection.get(limit=100)
        print("DOCUMENTS", collection_metadatas['documents'])
        print("\n\n\n ids",collection_metadatas['ids'])


if __name__ == '__main__':
    unittest.main()
