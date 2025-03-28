import numpy as np
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from utils.dataset import cosine_similarity_search


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve_csv_database_cross_titles_kws(state: State, vector_store_titles, vector_store_keywords):

    retrieved_docs_titles = vector_store_titles.similarity_search_with_score(state["question"], k=5)
    retrieved_docs_keywords = vector_store_keywords.similarity_search_with_score(state["question"], k=5)

    title_scores = {doc.metadata["document"]: score for doc, score in retrieved_docs_titles}
    keyword_scores = {doc.metadata["document"]: score for doc, score in retrieved_docs_keywords}

    title_pg_contents = {doc.metadata["document"]: doc.page_content for doc, _ in retrieved_docs_titles}
    keyword_pg_contents = {doc.metadata["document"]: doc.page_content for doc, _ in retrieved_docs_keywords}

    common_docs = []
    avg_scores = []
    for doc_id in title_scores.keys() & keyword_scores.keys():  # Intersection of doc IDs
        avg_score = (title_scores[doc_id] + keyword_scores[doc_id]) / 2.  # Averaging scores
        common_docs.append(doc_id)
        avg_scores.append(avg_score)
    # Sort by best (lowest) average score
    # TODO: check if the lowest score is actually the best
    avg_scores.sort()
    sorted_indices = sorted(range(len(avg_scores)), key=lambda ii: avg_scores[ii])
    common_docs = [common_docs[ii] for ii in sorted_indices]

    output_documents = []

    for ii, doc_id in enumerate(common_docs):
        output_documents.append(
            (Document(
                id='', metadata={'document': doc_id},
                # page_content=keyword_pg_contents.get(doc_id)
                page_content=title_pg_contents.get(doc_id)
            ),
             avg_scores[ii])
        )

    # TODO: this may be a problem because the output_documents are not recognized
    #  by the generate even though they should be correct

    return {"context": output_documents}

def retrieve_csv_database(state: State, vector_store_titles, vector_store_keywords=None):

    retrieved_docs_titles = vector_store_titles.similarity_search_with_score(state["question"], k=5)
    # retrieved_docs_keywords = [vector_store_keywords[ii] for ii in sorted_indices]
    # TODO: need to implement the selection of the found vector_store_keywords for the same indexes in retrieved_doc_titles

    return {"context": retrieved_docs_titles}

def retrieve_sql_database(state: State, vector_store):
    retrieved_docs = vector_store.similarity_search_with_score(state["question"], k=5)
    return {"context": retrieved_docs}


def generate(state: State, prompt, llm):
    # doc[0] because the documents in state are returned from a similarity_search_w_score,
    # ergo a tuple of (doc, score)
    docs_content = "\n\n".join(doc[0].page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def generate_synonyms(prompt, llm, query):
    messages = prompt.invoke({"name": "Waggy", "context": query})
    response = llm.invoke(messages)
    return response.content

def find_semantically_similar_text(query: str, llm: ChatMistralAI):

    prompt = ChatPromptTemplate([
        ("system", "<s> [INST] You are a helpful AI bot. Your name is {name}. Given the following sentence: {context}, "
                   "identify and list synonyms or semantically similar sentences. "
                   "Ensure the suggested sentences maintain the context and meaning of the original sentence. "
                   "Please write maximum five sentences in your answer separated by commas. [/INST] </s> \n[INST] Answer: [/INST]")
    ])

    synonyms = generate_synonyms(prompt=prompt, llm=llm, query=query)

    return synonyms

def similarity_search(documents, field_name: str, query: str | list):

    # TODO: if the field is repo_name, one should split the query in different words and do the similarity search with the words,
    #  because the repository names are never longer than one word.

    documents_list = []
    urls_list = []

    for doc in documents:
        documents_list.append(
            Document(
                id='', metadata={'document': doc.page_content.split('paper_identifier: ')[1].split('\n')[0]},
                # provides the document id in the metadata
                page_content=doc.page_content.split(f'{field_name}: ')[1].split('\n')[0]
            )
        )
        if field_name in ["repo_name", "repo_description"]:
            urls_list.append(doc.page_content.split('url: ')[1].split('\n')[0])

    document_list_texts = [document.page_content for document in documents_list]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(document_list_texts)
    if isinstance(query, str):
        if urls_list:
            similar_documents_list, cosine_sim_score, urls_of_similar_documents = cosine_similarity_search(
                documents=documents_list, query=query, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix, urls=urls_list
            )
        else:
            similar_documents_list, cosine_sim_score = cosine_similarity_search(
                documents=documents_list, query=query, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix, urls=urls_list
            )
    elif isinstance(query, list):
        doc_lists_to_filter = []
        score_lists_to_filter = []
        url_lists_to_filter = []
        for qq in query:
            assert isinstance(qq, str), "query can only be a string."
            if urls_list:
                docs, scores, urls = cosine_similarity_search(
                    documents=documents_list, query=qq, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix, urls=urls_list
                )
                url_lists_to_filter.extend(urls)
            else:
                docs, scores = cosine_similarity_search(
                    documents=documents_list, query=qq, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix, urls=urls_list
                )
            doc_lists_to_filter.extend(docs)
            score_lists_to_filter.extend(scores)
        # getting the top ten best scoring documents
        sorted_indexes_from_scores = np.argsort(score_lists_to_filter).tolist()
        sorted_documents = [doc_lists_to_filter[ii] for ii in sorted_indexes_from_scores]
        filtered_docs, indices = np.unique([doc.page_content for doc in sorted_documents], return_index=True)  # removing repetitions
        score_lists = [score_lists_to_filter[ii] for ii in indices]  # indices of the unique documents
        top_ten = sorted(enumerate(score_lists), key=lambda x: x[1], reverse=True)[:10]  # keeping only the top ten scores
        top_ten_indices = [index for index, _ in top_ten]
        cosine_sim_score = [score for _, score in top_ten]  # cosine sim search scores for the top ten
        similar_documents_list = [doc_lists_to_filter[ii] for ii in top_ten_indices]  # documents for the top ten
        if urls_list:
            urls_of_similar_documents = [url_lists_to_filter[ii] for ii in top_ten_indices]
    else:
        raise TypeError("query can be only a string or a list of strings.")

    if urls_list:
        return similar_documents_list, cosine_sim_score, urls_of_similar_documents

    else:
        return similar_documents_list, cosine_sim_score


def similarity_search_hess_pages(documents_texts: list, query: str | list):
    """

    :param query: the query to be searched for inside the database
    :param documents_texts: documents texts from the SQL database, already filtered for the chosen section
    :return:
    """

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents_texts)

    if isinstance(query, str):
        docs_output, scores_output, indexes_output = cosine_similarity_search(
            documents=documents_texts, query=query,
            vectorizer=vectorizer, tfidf_matrix=tfidf_matrix, urls=[], top_n=5, return_indexes=True
        )
    elif isinstance(query, list):
        doc_list_to_filter = []
        score_list_to_filter = []
        indexes_list_to_filter = []
        for qq in query:
            assert isinstance(qq, str), "query can only be a string."
            docs, scores, indexes = cosine_similarity_search(
                documents=documents_texts, query=qq,
                vectorizer=vectorizer, tfidf_matrix=tfidf_matrix, urls=[], top_n=10, return_indexes=True
            )
            doc_list_to_filter.extend(docs)
            score_list_to_filter.extend(scores)
            indexes_list_to_filter.extend(indexes)

        sorted_indexes_from_scores = np.argsort(score_list_to_filter).tolist()
        sorted_documents = [doc_list_to_filter[ii] for ii in sorted_indexes_from_scores]
        filtered_docs, indices = np.unique([document for document in sorted_documents], return_index=True)  # removing repetitions
        score_lists = [score_list_to_filter[ii] for ii in indices]  # indices of the unique documents
        top_ten = sorted(enumerate(score_lists), key=lambda x: x[1], reverse=True)[:10]  # keeping only the top ten scores
        top_ten_indices = [index for index, _ in top_ten]
        cosine_sim_score = [score for _, score in top_ten]  # cosine sim search scores for the ten five

        scores_output = cosine_sim_score
        docs_output = [doc_list_to_filter[ii] for ii in top_ten_indices]
        indexes_output = [indexes_list_to_filter[ii] for ii in top_ten_indices]
    else:
        raise TypeError("query can be only a string or a list of strings.")

    return docs_output, scores_output, indexes_output