# -*- coding: utf-8 -*-
"""

This script is executable with python -m run.dataset_creation.merge_url_databases
"""
import pandas as pd
import numpy as np
from glob import glob


def main() -> None:

    list_paths_metadata = glob("./data/database_with_urls/*metadata*")
    list_paths_urls = glob("./data/database_with_urls/*pdf*")

    year_month_combinations = [path.split("clean_")[1].split(".txt")[0] for path in list_paths_urls]
    year_month_combinations.sort()
    years = np.unique([year_month_combination[:2] for year_month_combination in year_month_combinations]).astype(int).tolist()
    print("These are the available years: ")
    for ii, year in enumerate(years):
        print(ii, "20" + str(year) if year < 25 else "19" + str(year))
    chosen_index = int(
        input("Please choose for which year you want me to merge the datasets by typing the index and ENTER. "
              "If you want me to merge all the years type -1 and ENTER. ")
    )
    if 0 < chosen_index < len(years):
        chosen_year = str(years[chosen_index])
        year_month_combinations = [year_month_combination for year_month_combination in year_month_combinations if chosen_year in year_month_combination]
    elif chosen_index == -1:
        year_month_combinations = year_month_combinations
    else:
        raise ValueError("Wrong input index.")

    print(year_month_combinations)

    string_range = year_month_combinations[0] + "-" + year_month_combinations[-1]

    df_urls_complete = pd.DataFrame()
    df_metadata_complete = pd.DataFrame()

    for combination in year_month_combinations:
        year = "20" + combination[:2]
        month = combination[2:]

        mask_year_month = [combination in s for s in list_paths_urls]
        filtered_list_paths_urls = [s for s, mask in zip(list_paths_urls, mask_year_month) if mask]

        substring = year + "-" + month
        mask_year_month = [substring in s for s in list_paths_metadata]
        filtered_list_paths_metadata = [s for s, mask in zip(list_paths_metadata, mask_year_month) if mask]

        df_urls = pd.read_csv(filtered_list_paths_urls[0], sep='\t', names=["identifier", "url"], on_bad_lines='skip', keep_default_na=False)
        df_metadata = pd.read_csv(filtered_list_paths_metadata[0], header=0, sep='\t', on_bad_lines='skip')

        df_urls_complete = pd.concat([df_urls_complete, df_urls])
        df_metadata_complete = pd.concat([df_metadata_complete, df_metadata])

        print(combination, len(df_urls_complete), len(df_metadata_complete))

    df_urls_complete.to_csv("./data/database_with_urls/complete/database_urls_"+string_range+".csv", sep="\t")
    print("Saved DataFrame with urls at " + "./data/database_with_urls/complete/database_urls_"+string_range+".csv")
    df_metadata_complete.to_csv("./data/database_with_urls/complete/database_metadata_"+string_range+".csv", sep="\t")
    print("Saved DataFrame with metadata at " + "./data/database_with_urls/complete/database_metadata_" + string_range + ".csv")

if __name__ == '__main__':

    main()