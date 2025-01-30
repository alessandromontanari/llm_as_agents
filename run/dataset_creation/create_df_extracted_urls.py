# -*- coding: utf-8 -*-
"""

This script is executable with python -m run.dataset_creation.create_df_extracted_urls
"""
from utils.dataset import PaperAbstractsDatasetGeneration
# TODO: implement logging


def main() -> None:

    dataset_generator = PaperAbstractsDatasetGeneration(
        path_in_csv="./data/database_with_urls/database_metadata_short_2019-01-31.csv",
        path_in_url_file="./data/database_with_urls/database_pdf_clean_1901.txt",
        skip_rows=0,
        max_num_abstracts=None,
        mask_astro_ph=True,
        mask_hess=False,
        save_dataframe_as_json=False,
        save_dataframe_raw_texts=False,
        save_dataframe_code_mentions=False,
        save_dataframe_cross_checked_code_mentions=True
    )

if __name__ == '__main__':

    main()