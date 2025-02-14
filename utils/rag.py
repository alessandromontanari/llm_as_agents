from langchain_core.documents import Document
from typing_extensions import List, TypedDict


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
    # doc[0] because the documents in state are returned from a similarity_search_w_score, ergo a tuple of (doc, score)
    docs_content = "\n\n".join(doc[0].page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}