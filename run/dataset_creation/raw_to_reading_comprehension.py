# -*- coding: utf-8 -*-
"""

This script is executable with python -m run.dataset_creation.raw_to_reading_comprehension
"""
import argparse
from tqdm.contrib.concurrent import process_map
from utils.read import TYPES, type_map, get_max_workers
from pysbd import Segmenter
import copy
import functools
import pandas as pd

# TODO: need to understand if sentenpiece is needed...

def search(entry_dataset, overall_class_type, text_segmenter, initialized_type_map, arguments):
    title = entry_dataset['text'].split('\n')[0]
    context_wo_title = '\n'.join(entry_dataset['text'].split('\n')[1:])

    context_wo_title = overall_class_type.truncate_sentence(context_wo_title, max_len=overall_class_type.max_seq_len - 200)

    # mine task examples from the raw text
    sentences = text_segmenter.segment(context_wo_title)
    overall_entry = {'text_id': entry_dataset['text_id']}
    for available_type_class in TYPES:
        type_class = initialized_type_map[available_type_class]
        overall_entry[available_type_class], mined_num = type_class.mine(
            text=context_wo_title,
            domain=arguments.domain_name,
            title=title,
            sents=copy.deepcopy(sentences)
        )

    # create the reading comprehension text
    reading_comprehension_texts, reading_comprehension_qa_separate, count_dict = overall_class_type.format_recomprehension(
        copy.deepcopy(overall_entry)
    )
    # count_dict includes the number of comprehension tasks per task type
    # you may use `mined_num` and `count_dict` for data analysis

    return {
        'reading_comprehension': reading_comprehension_texts,
        'reading_comprehension_qa': reading_comprehension_qa_separate,
        'file_name': entry_dataset['file_name']
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset',
                        type=str, help='directory of the input raw texts',
                        default='./data/dataset/dataset_raw_texts_astroph1_hess0_skiprows0_maxrows19648.csv')
    parser.add_argument('--output_dir',
                        type=str, help='directory of the output reading comprehension texts',
                        default='./data/dataset_reading_comprehension/')
    parser.add_argument("--general_spm_path",
                        type=str, help='path to the sentencepiece model of the general LLM',
                        default='./models/sentencepiece/astro-ph.model')  # TODO: this should be modified to the general text spm file
    parser.add_argument("--domain_spm_path",
                        type=str, help='path to the sentencepiece model trained from the target domain corpora',
                        default='./models/sentencepiece/astro-ph.model')
    parser.add_argument("--domain_name",
                        type=str, help='target domain name, e.g., `physics`, `mathematics`',
                        default='astro-ph')
    # TODO: to be modified with what we get from the arxiv

    args = parser.parse_args()

    # get max worker for multiprocess
    max_workers = get_max_workers()
    print(f'max_workers: {max_workers}')

    # load from the input dataset
    print('loading raw texts from the input dataset...')
    input_dataset = pd.read_csv(args.input_dataset, header=0)

    raw_texts = []
    for index, row in input_dataset.iterrows():
        raw_texts.append({'text': row["raw_texts"], 'text_id': index, 'file_name': args.input_dataset + f", line_{index}"})

    # init type_map
    inited_type_map = {}
    for available_type in TYPES:
        type_cls = type_map.cls_dic[available_type]()
        type_cls.init_spm(args.general_spm_path, args.domain_spm_path)
        inited_type_map[available_type] = type_cls

    overall_cls = type_map.cls_dic['overall']()
    overall_cls.init_spm(args.general_spm_path, args.domain_spm_path)

    # to chunk text to sentences
    segmenter = Segmenter(language='en', clean=False)

    partial_search = functools.partial(search, overall_class_type=overall_cls, text_segmenter=segmenter,
                                       initialized_type_map=inited_type_map, arguments=args)
    print('transferring raw texts into reading comprehension...')
    reading_comprehension = list(process_map(partial_search, raw_texts, max_workers=max_workers, chunksize=8192))

    reading_comprehension_output = []
    reading_comprehension_q_output = []
    reading_comprehension_a_output = []

    for el in reading_comprehension:
        reading_comprehension_output.append(el['reading_comprehension'])
        reading_comprehension_q_output.append(el['reading_comprehension_qa'][0])
        reading_comprehension_a_output.append(el['reading_comprehension_qa'][1])

    input_dataset['reading_comprehension_texts'] = reading_comprehension_output
    input_dataset['reading_comprehension_texts_q'] = reading_comprehension_q_output
    input_dataset['reading_comprehension_texts_a'] = reading_comprehension_a_output

    file_name = "dataset_reading_comprehension" + args.input_dataset.split("raw_texts")[1]

    input_dataset.to_csv(
        args.output_dir + file_name,
        index=False
    )

    print(f'saved to {args.output_dir}')
