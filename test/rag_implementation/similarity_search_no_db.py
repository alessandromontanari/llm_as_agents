import unittest

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
import pandas as pd

def cosine_similarity_search(documents, query, vectorizer, tfidf_matrix, top_n=10):

    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    most_similar_indices = np.argsort(cosine_similarities)[-top_n:][::-1]

    return [documents[ii] for ii in most_similar_indices]


class TestSimilaritySearchNoDB(unittest.TestCase):
    def test_similarity_search_no_db(self):

        df = pd.read_csv(
            "./data/dataset/dataset_cross_checked_code_mentions_astroph1_hess0_skiprows0_maxrows60569.csv",
            delimiter=",",
            header=0
        )

        # create documents for titles
        metadata_doc = [{"document": nn} for nn in range(len(df["title"]))]

        splitter = CharacterTextSplitter(
            chunk_size=100,  # max length for each chunk
            chunk_overlap=0,  # up to ten chars overlapping between chunks
            length_function=len,
            is_separator_regex=False,
            separator='\n'
        )

        documents_title = splitter.create_documents(df["title"], metadatas=metadata_doc)

        texts_title = [document.page_content for document in documents_title]

        # Vectorize the texts using TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts_title)

        # Perform a cosine similarity search
        query = "Black hole astronomy"
        similar_documents = cosine_similarity_search(documents_title, query, vectorizer, tfidf_matrix)

        # Print the results
        for doc in similar_documents:
            print(doc.page_content)


if __name__ == '__main__':
    unittest.main()
