import os
from dotenv import load_dotenv
load_dotenv()

os.getenv("MISTRAL_API_KEY")
os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_API_KEY")

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import unittest


class TestComponentsInitialization(unittest.TestCase):
    def test_components_initialization(self):

        llm = ChatMistralAI(model="mistral-large-latest")
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        vector_store = InMemoryVectorStore(embeddings)

        # may think of a few assertions to make these units work

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
