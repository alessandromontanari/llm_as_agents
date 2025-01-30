# -*- coding: utf-8 -*-
"""

This script is executable with python -m run.dataset_creation.create_df_raw_text_format
"""
from utils.dataset import PaperAbstractsDatasetGeneration
# TODO: implement logging


def main() -> None:

    dataset_generator = PaperAbstractsDatasetGeneration(
        path_in_csv="./data/database/abstracts.csv",
        skip_rows=0,
        max_num_abstracts=None,
        mask_astro_ph=True,
        mask_hess=False,
        save_dataframe_as_json=False,
        save_dataframe_raw_texts=True,
    )


if __name__ == '__main__':

    main()