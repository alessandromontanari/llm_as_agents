import os
from dotenv import load_dotenv
load_dotenv()

os.getenv("MISTRAL_API_KEY")
os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_API_KEY")

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain.text_splitter import CharacterTextSplitter
from langchain import hub
from langchain_chroma import Chroma
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
import unittest
import pandas as pd


class TestQueryImplementation(unittest.TestCase):
    def test_query_implementation(self):

        llm = ChatMistralAI(model="open-mistral-7b")
        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            wait_time=40,  # Should be used for rate limit retries
            max_retries=5,
            timeout=300
        )

        df = pd.read_csv(
            "./data/dataset/dataset_cross_checked_code_mentions_astroph1_hess0_skiprows0_maxrows337.csv",
            delimiter=",",
            header=0
        )

        titles = df["title"]

        # create documents for titles
        metadata_doc = [{"document": nn} for nn in range(len(titles))]

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
            persist_directory="./data/vector_stores/chroma_langchain_db_test",
            # Where to save data locally, remove if not necessary
            collection_metadata=None,  # one can add metadatas, Dict
        )

        prompt = hub.pull("rlm/rag-prompt")

        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str

        def retrieve(state: State):
            retrieved_docs = vector_store.similarity_search_with_score(state["question"], k=5)
            return {"context": retrieved_docs}

        def generate(state: State):
            docs_content = "\n\n".join(doc[0].page_content for doc in state["context"])
            messages = prompt.invoke({"question": state["question"], "context": docs_content})
            response = llm.invoke(messages)
            return {"answer": response.content}

        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        question = "Can you find anything among the provided titles about Cherenkov astronomy?"

        result = graph.invoke({
            "question": question}
        )

        print("You've asked: \n" + question)

        identified_urls = []

        print("\nHere are the first five matching titles:")
        for context_title in result["context"]:
            row = df.loc[df["title"] == context_title[0].page_content]
            identified_urls.append(row["urls"])

            print(f"{context_title[0].page_content:<100} score: {context_title[1]}")

        print(f'\nAnswer to your question: {result["answer"]}')

        print(identified_urls)

if __name__ == '__main__':
    unittest.main()
