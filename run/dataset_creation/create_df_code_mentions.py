# -*- coding: utf-8 -*-
"""

This script is executable with python -m run.dataset_creation.create_df_code_mentions
"""
from utils.dataset import PaperAbstractsDatasetGeneration
# TODO: implement logging


def main() -> None:

    dataset_generator = PaperAbstractsDatasetGeneration(
        path_in_csv="./data/database/abstracts.csv",
        path_in_url_file=None,
        skip_rows=0,
        max_num_abstracts=None,
        mask_astro_ph=False,
        mask_hess=False,
        save_dataframe_as_json=False,
        save_dataframe_raw_texts=False,
        save_dataframe_code_mentions=True
    )

if __name__ == '__main__':

    main()