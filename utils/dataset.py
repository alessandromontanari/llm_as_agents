import numpy as np
import pandas as pd
import torch
import more_itertools
import os
import re
import yake

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset


class PaperAbstractsDatasetGeneration:
    def __init__(
            self,
            path_in_url_file: str = None,
            path_in_csv: str = None,
            skip_rows: int = 0,
            max_num_abstracts: int = None,
            mask_astro_ph: bool = False,
            mask_hess: bool = False,
            save_dataframe_as_json: bool = False,
            save_dataframe_raw_texts: bool = False,
            save_dataframe_code_mentions: bool = False,
            save_dataframe_cross_checked_code_mentions: bool = False,
    ):
        """

        :param path_in_url_file: str, path to the database of cleaned urls.
        :param path_in_csv: str, to the database of inputs.
        :param skip_rows: int, how many rows to skip from the head of the database.
        :param max_num_abstracts: int, how many rows to be included. Each row consists of an abstract.
        :param mask_astro_ph: bool, whether to consider only abstracts classified as astro-ph
        :param mask_hess: bool, whether to consider only papers including H.E.S.S. in the abstract
        :param save_dataframe_as_json: bool, whether to save the dataset as a jsonl file.
        :param save_dataframe_raw_texts: bool, whether to save the dataset of titles + raw texts.
        :param save_dataframe_code_mentions: bool, whether to save the dataset containing the software mentions.
        :param save_dataframe_cross_checked_code_mentions: bool,
        """

        self.save_dataframe_code_mentions = save_dataframe_code_mentions
        self.url_file_path = path_in_url_file
        self.csv_path = path_in_csv if path_in_csv else "./data/database/abstracts.csv"

        self.mask_astro_ph = mask_astro_ph
        self.mask_hess = mask_hess

        self.skip_rows = skip_rows
        self.max_num_abstracts = max_num_abstracts

        self.dataframe_abstracts = self.load_data()

        # TODO: there may be a problem here with the extraction because for the different functions there are different exceptions thrown,
        #  most of the times not corresponding
        self.list_queries = self.extract_queries()

        self.list_sources = self.extract_sources()

        self.dict_code_mentions = self.extract_code_mentions()

        if self.url_file_path is not None:
            self.dataframe_urls = self.load_data_urls()
            self.dict_cross_checked_code_mentions = self.cross_check_extract_code_mentions()

        # TODO: this may become available with a bool option
        self.list_bodies = self.extract_bodies()
        self.list_titles = self.extract_titles()

        if save_dataframe_as_json:
            self.dataframe_prompt_response_source = self.create_dataframe_prompt_response_source()
            self.save_dataframe_prompt_response_source_as_json()

        if save_dataframe_raw_texts:
            self.dataframe_raw_texts_format = self.create_dataframe_raw_texts_format()
            self.save_dataframe_raw_texts_format_as_csv()

        if save_dataframe_code_mentions:
            self.dataframe_code_mentions = self.create_dataframe_code_mentions()
            self.save_dataframe_code_mentions_as_csv()

        if save_dataframe_cross_checked_code_mentions:
            self.dataframe_cross_checked_code_mentions = self.create_dataframe_cross_checked_code_mentions()
            self.save_dataframe_cross_checked_code_mentions_as_csv()

    def load_data(self) -> pd.DataFrame:

        dataframe = pd.read_csv(
            self.csv_path,
            header=0,
            sep='\t',
            skiprows=self.skip_rows,
            nrows=self.max_num_abstracts,
            on_bad_lines='skip',
        )

        if self.mask_astro_ph:
            astro_ph_mask = dataframe['category'] == 'physics:astro-ph'
            dataframe_masked = dataframe[astro_ph_mask]
            if self.mask_hess:
                hess_mask = dataframe_masked['description'].str.contains('H.E.S.S.')
                dataframe_out = dataframe_masked[hess_mask]
            else:
                dataframe_out = dataframe_masked
        else:
            dataframe_out = dataframe

        return dataframe_out

    def load_data_urls(self):

        dataframe = pd.read_csv(
            self.url_file_path,
            # names=["identifier", "url"],
            header=0,
            sep='\t',
            skiprows=self.skip_rows,
            nrows=self.max_num_abstracts,
            on_bad_lines='skip',
            keep_default_na=False
        )

        # some identifiers may be ill-defined...
        mask_identifiers = dataframe["identifier"].str.isdigit()
        dataframe = dataframe[mask_identifiers]

        # TODO: not yet implemented the astro-ph and H.E.S.S. masks here,
        #  this dataframe may not be of the same length as the one from load_data...

        return dataframe

    def extract_queries(self) -> list:

        print("Extracting queries...")

        contexts_out = []

        for index, content in tqdm(enumerate(self.dataframe_abstracts["description"]), total=len(self.dataframe_abstracts["description"])):

            try:
                context_out = []
                to_append = f'''
                This is the context
                Context: {content.split(";Comment")[0]}
    
                Please answer the following question by extracting parts of the context and don't elaborate on them.
                '''

                context_out.append(to_append + "\nQuestion: What region of the sky was or will be observed for this study?")
                context_out.append(to_append + "\nQuestion: Which data -- and only data, not models -- is used for this analysis?")
                context_out.append(to_append + "\nQuestion: Which telescopes were used or will be used to observe to collect data?")

                contexts_out.append(context_out)
            except Exception as e:
                print(f"Skipping {self.dataframe_abstracts["identifier"][index]} because badly formatted, error {e}.")
                continue

        return contexts_out

    def extract_bodies(self) -> list:

        print("Extracting abstract bodies...")

        bodies_out = []

        for content in tqdm(self.dataframe_abstracts["description"], total=len(self.dataframe_abstracts["description"])):

            try:
                bodies_out.append(content.split(";Comment")[0])
            except Exception as e:
                print(f"Skipping {content} because badly formatted, error {e}.")
                continue

        return bodies_out

    def extract_sources(self):

        print("Extracting abstract sources...")

        sources = self.dataframe_abstracts[['title', 'link', 'creator', 'dates']]

        sources_out = []

        for index, source in tqdm(sources.iterrows(), total=len(sources)):
            try:
                sources_out.append(source['title'] + '\n' + source['link'] + '\n' + source['creator'] + '\n' + source['dates'])
            except Exception as e:
                print(f"Skipping {source['title']} because badly formatted, error {e}.")
                continue

        return sources_out

    def extract_titles(self):

        print("Extracting titles...")

        sources = self.dataframe_abstracts[['title', 'link', 'creator', 'dates']]

        titles_out = []

        for index, source in tqdm(sources.iterrows(), total=len(sources)):
            titles_out.append(source['title'])

        return titles_out

    def extract_code_mentions(self) -> dict:

        print("Extracting where code is mentioned...")

        sources = self.dataframe_abstracts[['title', 'description', 'category']]
        titles, keywords, categories, mentioned_software, urls = [], [], [], [], []

        for index, source in tqdm(sources.iterrows(), total=len(sources)):

            try:
                if "code" in source['description']:

                    pattern_code = r"([^.?!]*\b(?:code|software)\b[^.?!]*[.?!])"
                    matches_code = re.findall(pattern_code, source['description'].split(";Comment")[0], flags=re.IGNORECASE)

                    pattern_kw = r"(?:Keywords|KeyWords|keywords):\s*(.+?)(?:[.;]|$)"
                    match_kw = re.search(pattern_kw, source['description'].split(";Comment")[0])

                    pattern_url =  re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
                    matches_url = pattern_url.findall(source['description'].split(";Comment")[0])

                    if match_kw and matches_code and matches_url:
                        titles.append(source['title'])
                        keywords.append(match_kw.group(1).strip())
                        mentioned_software.append(matches_code)
                        urls.append(matches_url)
                        categories.append(source['category'])
                    elif matches_code and matches_url:
                        titles.append(source['title'])
                        keywords.append('')
                        mentioned_software.append(matches_code)
                        urls.append(matches_url)
                        categories.append(source['category'])
                    elif matches_url:
                        titles.append(source['title'])
                        keywords.append('')
                        mentioned_software.append('')
                        urls.append(matches_url)
                        categories.append(source['category'])
                    else:
                        titles.append(source['title'])
                        keywords.append('')
                        mentioned_software.append('')
                        urls.append('')
                        categories.append(source['category'])
            except Exception as e:
                print(f"Skipping {index} because badly formatted, error {e}")
                continue

        return {'title': titles, 'keywords': keywords, 'category': categories, 'mentioned_software': mentioned_software, 'urls': urls}

    def intersect_identifiers(self, df_urls, df_sources):

        identifiers_from_urls = df_urls["identifier"].to_numpy()
        identifiers_from_urls = [int(identifier) for identifier in identifiers_from_urls if identifier.isdigit()]
        identifiers_from_sources = df_sources["identifier"].to_numpy()
        identifiers_from_sources = [int(identifier) for identifier in identifiers_from_sources if "/" not in identifier]

        intersection_urls = np.intersect1d(ar1=identifiers_from_urls, ar2=identifiers_from_sources)

        mask_identifiers_urls = np.isin(identifiers_from_urls, intersection_urls)

        return df_urls[mask_identifiers_urls]

    def cross_check_extract_code_mentions(self):

        df_urls = self.dataframe_urls
        df_sources = self.dataframe_abstracts[['identifier', 'title', 'description', 'category']]

        df_urls = self.intersect_identifiers(df_urls, df_sources)

        print("Extracting and cross-checking software urls and the metadata of the corresponding identifier...")

        # settings for yake keyword extractor
        language = "en"
        max_ngram_size = 3

        kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, top=6)

        identifiers, titles, keywords, categories, mentioned_software, output_urls = [], [], [], [], [], []

        df_urls_identifiers_list = list(map(int, df_urls["identifier"].tolist()))

        for index, source in tqdm(df_sources.iterrows(), total=len(df_sources)):

            identifier = source['identifier']

            abstract = source['description'].split(";Comment")[0]

            try:
                if int(identifier) in np.array(df_urls_identifiers_list):
                    pattern_code = r"([^.?!]*\b(?:code|software)\b[^.?!]*[.?!])"
                    matches_code = re.findall(pattern_code, source['description'].split(";Comment")[0], flags=re.IGNORECASE)
                    pattern_kw = r"(?:Keywords|KeyWords|keywords):\s*(.+?)(?:[.;]|$)"
                    match_kw = re.search(pattern_kw, source['description'].split(";Comment")[0])

                    keywords_to_append = kw_extractor.extract_keywords(abstract)
                    keywords_to_append = [kw[0] for kw in keywords_to_append]

                    identifiers.append(int(identifier))
                    titles.append(source['title'])
                    categories.append(source['category'])
                    output_urls.append(df_urls[df_urls["identifier"] == identifier]["url"].values)
                    if match_kw and matches_code:
                        keywords.append(match_kw.group(1).strip())
                        mentioned_software.append(matches_code)
                    elif matches_code:
                        keywords.append(keywords_to_append)
                        mentioned_software.append(matches_code)
                    else:
                        keywords.append(keywords_to_append)
                        mentioned_software.append('')
            except Exception as e:
                continue

        return {'identifier': identifiers, 'title': titles, 'keywords': keywords, 'category': categories,
                'mentioned_software': mentioned_software, 'urls': output_urls}

    def create_dataframe_prompt_response_source(self) -> pd.DataFrame:

        list_prompts = self.list_queries
        list_sources = self.list_sources

        dataframe_out = pd.DataFrame(columns=['prompt', 'response', 'source'])

        for index, prompts in enumerate(list_prompts):
            for prompt in prompts:
                dataframe_out.loc[len(dataframe_out)] = [prompt, '', list_sources[index]]

        return dataframe_out

    def create_dataframe_raw_texts_format(self):

        list_bodies = self.list_bodies
        list_titles = self.list_titles
        list_identifiers = self.dataframe_abstracts['identifier'].tolist()

        dataframe_out = pd.DataFrame(columns=['identifier', 'raw_texts'])

        for index, body in enumerate(list_bodies):
            dataframe_out.loc[len(dataframe_out)] = [list_identifiers[index], list_titles[index] + '\n' + body]

        return dataframe_out

    def create_dataframe_code_mentions(self):

        return pd.DataFrame(self.dict_code_mentions)

    def create_dataframe_cross_checked_code_mentions(self):

        return pd.DataFrame(self.dict_cross_checked_code_mentions)

    def save_dataframe_prompt_response_source_as_json(self):

        path_to_dir = "./data/dataset/"
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)

        pyarrow_dataset = Dataset.from_pandas(df=self.dataframe_prompt_response_source, preserve_index=False)

        name_save_file = path_to_dir + (
            f"dataset_pyarrow_astroph{int(self.mask_astro_ph)}"
            f"_hess{int(self.mask_hess)}"
            f"_skiprows{self.skip_rows}"
            f"_maxrows{self.max_num_abstracts}.jsonl") if self.max_num_abstracts is not None else path_to_dir + (
            f"dataset_pyarrow_astroph{int(self.mask_astro_ph)}"
            f"_hess{int(self.mask_hess)}"
            f"_skiprows{self.skip_rows}"
            f"_maxrows{len(self.dataframe_prompt_response_source)}.jsonl"
        )

        pyarrow_dataset.to_json(name_save_file)

    def save_dataframe_raw_texts_format_as_csv(self):

        path_to_dir = "./data/dataset/"
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)

        name_save_file = path_to_dir + (
            f"dataset_raw_texts_astroph{int(self.mask_astro_ph)}"
            f"_hess{int(self.mask_hess)}"
            f"_skiprows{self.skip_rows}"
            f"_maxrows{self.max_num_abstracts}.csv") if self.max_num_abstracts is not None else path_to_dir + (
            f"dataset_raw_texts_astroph{int(self.mask_astro_ph)}"
            f"_hess{int(self.mask_hess)}"
            f"_skiprows{self.skip_rows}"
            f"_maxrows{len(self.dataframe_raw_texts_format)}.csv"
        )

        self.dataframe_raw_texts_format.to_csv(name_save_file, index=True)

    def save_dataframe_code_mentions_as_csv(self):

        path_to_dir = "./data/dataset/"
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)

        name_save_file = path_to_dir + (
            f"dataset_code_mentions_astroph{int(self.mask_astro_ph)}"
            f"_hess{int(self.mask_hess)}"
            f"_skiprows{self.skip_rows}"
            f"_maxrows{self.max_num_abstracts}.csv") if self.max_num_abstracts is not None else path_to_dir + (
            f"dataset_code_mentions_astroph{int(self.mask_astro_ph)}"
            f"_hess{int(self.mask_hess)}"
            f"_skiprows{self.skip_rows}"
            f"_maxrows{len(self.dataframe_code_mentions)}.csv"
        )

        self.dataframe_code_mentions.to_csv(name_save_file, index=True)

    def save_dataframe_cross_checked_code_mentions_as_csv(self):

        path_to_dir = "./data/dataset/"
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)

        name_save_file = path_to_dir + (
            f"dataset_cross_checked_code_mentions_astroph{int(self.mask_astro_ph)}"
            f"_hess{int(self.mask_hess)}"
            f"_skiprows{self.skip_rows}"
            f"_maxrows{self.max_num_abstracts}.csv") if self.max_num_abstracts is not None else path_to_dir + (
            f"dataset_cross_checked_code_mentions_astroph{int(self.mask_astro_ph)}"
            f"_hess{int(self.mask_hess)}"
            f"_skiprows{self.skip_rows}"
            f"_maxrows{len(self.dataframe_cross_checked_code_mentions)}.csv"
        )

        self.dataframe_cross_checked_code_mentions.to_csv(name_save_file, index=True)



def load_pyarrow_dataset(path_to_json_dataset: str, split_train_test: float = None):
    """

    :param path_to_json_dataset: path to the jsonl file save with PaperAbstractsDatasetGeneration.save_dataframe_prompt_response_source_as_json
    :param split_train_test: float, relative dimension of test to train split of the dataset. If None, no split is performed.
    :return: PyArrow dataset, if split_train_test is not None, the train/test split is also returned
    """

    if split_train_test is not None:
        # N.B., creating the dataset with Dataset.from_json() will output a Dataset
        pyarrow_dataset = Dataset.from_json(path_or_paths=path_to_json_dataset)
        pyarrow_dataset_dict_train_test = pyarrow_dataset.train_test_split(test_size=0.2)

        return pyarrow_dataset_dict_train_test

    else:
        # N.B., loading with load_dataset() outputs a DatasetDict
        pyarrow_dataset = load_dataset('json', data_files=path_to_json_dataset)
        return pyarrow_dataset

def cosine_similarity_search(documents, query, vectorizer, tfidf_matrix, top_n=10):

    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    most_similar_indices = np.argsort(cosine_similarities)[-top_n:][::-1]

    return [documents[ii] for ii in most_similar_indices], np.sort(cosine_similarities)[-top_n:][::-1]


class TrainingDatasetReader(torch.utils.data.Dataset):
    def __init__(
            self,
            model_name,
            dataset_path_and_name,
            test_dataset_size=0.2,
            cache_dir=None,
            max_length=2048,
            only_reading_comprehension=True,
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            model_max_length=max_length,  # specifies the maximum number of tokens the model can handle as input
            truncation_side="left",  # which side of the input sequence should be truncated if too long
            trust_remote_code=True,
        )

        self.inf_bsz = 16  # batch size

        self.tokenizer.padding_side = "right"  # Adds padding tokens to the end of the input sequence until it reaches the desired length

        self.only_reading_comprehension = only_reading_comprehension

        self.train_dataset, self.validation_dataset, self.test_dataset = self.get_dataset(dataset_path_and_name, test_dataset_size)
        # train/test split the dataset
        self.num_processes = 1
        self.process_index = 0


    def get_dataset(self, dataset_path_and_name, test_size):

        # TODO: this has to be modified depending on whether the training is for reading-comprehension or software mentions

        dataframe = pd.read_csv(
            dataset_path_and_name,
            names=['id', 'raw_texts', 'reading_comprehension_texts', 'reading_comprehension_texts_q', 'reading_comprehension_texts_a']
        )

        if self.only_reading_comprehension:
            dataframe = dataframe[['id', 'reading_comprehension_texts']]
            for index, row in dataframe.iterrows():
                row['reading_comprehension_texts'] = row['reading_comprehension_texts'].replace("\n", "[NEWLINE]")
                # this may only be needed for distil BERT cased, because it is otherwise stripping the newlines
        else:
            dataframe = dataframe[['id', 'reading_comprehension_texts_q', 'reading_comprehension_texts_a']]

        train_test_split = Dataset.from_pandas(dataframe).train_test_split(test_size=test_size, seed=42)
        train_validation_split = train_test_split["train"].train_test_split(test_size=0.25, seed=42)

        train_dataset = train_validation_split['train']
        validation_dataset = train_validation_split['test']
        test_dataset = train_test_split["test"]

        return train_dataset, validation_dataset, test_dataset

    def __getitem__(self, index):
        return self.train_dataset[index]

    def __len__(self):
        return len(self.train_dataset)

    def shard(self, accelerator):
        self.num_processes = accelerator.num_processes
        # number of processes being used for training or inference -> distributed compute nodes used for parallelism,
        # i.e., if there are 8 GPUs it will be 8
        self.process_index = accelerator.process_index
        # index of the current process
        # can be used to understand the role and identity of each process in distributed setups
        self.train_dataset = list(
            more_itertools.distribute(accelerator.num_processes, self.train_dataset)[accelerator.process_index]
        )