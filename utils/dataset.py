import pandas as pd
import torch
import more_itertools
import os
import re

from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset


class PaperAbstractsDatasetGeneration:
    def __init__(
            self,
            path_in: str = None,
            skip_rows: int = 0,
            max_num_abstracts: int = None,
            mask_astro_ph: bool = False,
            mask_hess: bool = False,
            save_dataframe_as_json: bool = False,
            save_dataframe_raw_texts: bool = False,
    ):
        """

        :param path_in: Path, to the database of inputs.
        :param skip_rows: int, how many rows to skip from the head of the database.
        :param max_num_abstracts: int, how many rows to be included. Each row consists of an abstract.
        :param mask_astro_ph: bool, whether to consider only abstracts classified as astro-ph
        :param mask_hess: bool, whether to consider only papers including H.E.S.S. in the abstract
        :param save_dataframe_as_json: bool, whether to save the dataset as a jsonl file.
        :param save_dataframe_raw_texts: bool, whether to save the dataset of titles + raw texts.
        """

        self.csv_path = path_in if path_in else "./data/abstract_database/abstracts.csv"

        self.mask_astro_ph = mask_astro_ph
        self.mask_hess = mask_hess

        self.skip_rows = skip_rows
        self.max_num_abstracts = max_num_abstracts

        self.dataframe_abstracts = self.load_data()

        # TODO: there may be a problem here with the extraction because for the different functions there are different exceptions thrown, most of the times not corresponding
        self.list_queries = self.extract_queries()

        self.list_sources = self.extract_sources()

        self.dict_code_mentions = self.extract_code_mentions()

        # TODO: this may become available with a bool option
        self.list_bodies = self.extract_bodies()
        self.list_titles = self.extract_titles()

        if save_dataframe_as_json:
            self.dataframe_prompt_response_source = self.create_dataframe_prompt_response_source()
            self.save_dataframe_prompt_response_source_as_json()

        if save_dataframe_raw_texts:
            self.dataframe_raw_texts_format = self.create_dataframe_raw_texts_format()
            self.save_dataframe_raw_texts_format_as_csv()


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

        print("Extracting where the code is mentioned...")

        sources = self.dataframe_abstracts[['title', 'description', 'category']]
        titles, keywords, categories, mentioned_software = [], [], [], []

        for index, source in tqdm(sources.iterrows(), total=len(sources)):

            try:
                if "code" in source['description']:

                    pattern_code = r"([^.?!]*\b(?:code|software)\b[^.?!]*[.?!])"
                    matches_code = re.findall(pattern_code, source['description'].split(";Comment")[0], flags=re.IGNORECASE)
                    pattern_kw = r"(?:Keywords|KeyWords|keywords):\s*(.+?)(?:[.;]|$)"
                    match_kw = re.search(pattern_kw, source['description'].split(";Comment")[0])
                    if match_kw and matches_code:
                        titles.append(source['title'])
                        keywords.append(match_kw.group(1).strip())
                        mentioned_software.append(matches_code)
                        categories.append(source['category'])
                    elif matches_code:
                        titles.append(source['title'])
                        keywords.append('')
                        mentioned_software.append(matches_code)
                        categories.append(source['category'])
                    else:
                        titles.append(source['title'])
                        keywords.append('')
                        mentioned_software.append('')
                        categories.append(source['category'])
            except Exception as e:
                print(f"Skipping {index} because badly formatted, error {e}")
                continue

        return {'title': titles, 'keywords': keywords, 'category': categories, 'mentioned_software': mentioned_software}

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

        dataframe_out = pd.DataFrame(columns=['raw_texts'])

        for index, body in enumerate(list_bodies):
            dataframe_out.loc[len(dataframe_out)] = [list_titles[index] + '\n' + body]

        return dataframe_out

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