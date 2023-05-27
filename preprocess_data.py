import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", type=Path)
    parser.add_argument("--patient_data_path", type=Path)

    args = parser.parse_args()
    return args

def remove_and_rename_columns(df):
    df = df[
        ["Patient ID", "Filename", "od", "os", \
        # EZ
        "fovea", "parafovea", "perifovea", "pericentral", \
            # RPE
            "fovea.1", "parafovea.1", "perifovea.1", "pericentral.1"]
        ]
    # Define the mapping of old column names to new column names
    column_mapping = {
        "fovea": "EZ-fovea",
        "parafovea": "EZ-parafovea",
        "perifovea": "EZ-perifovea",
        "pericentral": "EZ-pericentral",
        "fovea.1": "RPE-fovea",
        "parafovea.1": "RPE-parafovea",
        "perifovea.1": "RPE-perifovea",
        "pericentral.1": "RPE-pericentral"
    }

    # Rename the columns using the mapping
    df = df.rename(columns=column_mapping)
    return df

def do_sanity_check(df):
    expected_values = [-1, 0, 1]
    for col in df.columns:
        if col == "Patient ID" or col == "Filename":
            continue
        else:
            # only contain 0 or 1 or nan
            assert df[col].isin(expected_values).all()

def replace_special_value_in_label_col(df, col_name):
    # Define the pattern and replacement values
    pattern = r'^([0-1X])-.*'
    replacement = {'0': 0, '1': 1, 'X': 1}

    # Apply the regex pattern and replacement using the replace() method
    df[col_name] = df[col_name].str.replace(
        pattern, 
        lambda x: str(replacement.get(x.group(1), x.group(1))), 
        regex=True
        )
    
    df[col_name] = df[col_name].replace('X', 1)
    df[col_name] = df[col_name].fillna(-1).astype(int)
    return df

def fill_detailed_region(df):
    # Fill columns EZ-fovea, EZ-parafovea, EZ-perifovea, EZ-pericentral, RPE-fovea, RPE-parafovea, RPE-perifovea, RPE-pericentral to 0
    columns_to_fill = ['EZ-fovea', 'EZ-parafovea', 'EZ-perifovea', 'EZ-pericentral',
                       'RPE-fovea', 'RPE-parafovea', 'RPE-perifovea', 'RPE-pericentral']
    df[columns_to_fill] = df[columns_to_fill].fillna(0)

    return df

def preprocess_patient_data(df, patient_ids):
    # get only OCT data
    df = df[df["Exam type"] == "OCT"]

    # filter with unique patient ids
    df = df[df["Patient ID"].isin(patient_ids)]

    # keep od, os, EZ-fovea, EZ-parafovea, EZ-perifovea, EZ-pericentral, RPE-fovea, RPE-parafovea, RPE-perifovea, RPE-pericentral columns
    df = remove_and_rename_columns(df)

    # od, os: 0-XYZ -> 0, 1-XYZ -> 1, X-XYZ -> 1, nan -> -1
    df = replace_special_value_in_label_col(df, "od")
    df = replace_special_value_in_label_col(df, "os")

    # EZ-fovea, EZ-parafovea, EZ-perifovea, EZ-pericentral, RPE-fovea, RPE-parafovea, RPE-perifovea, RPE-pericentral
    df = fill_detailed_region(df)

    # Sanity check
    do_sanity_check(df)

    return df 

def preprocess_label_data(label_df):
    patient_with_labels = label_df[label_df["label"] == 1]
    # print(patient_with_labels.head())
    unique_patient_ids = patient_with_labels["Patient_ID"].unique()
    return patient_with_labels, unique_patient_ids

def main():
    args = parse_args()

    # Read the data
    label_df = pd.read_excel(args.label_path)
    patient_data_df = pd.read_excel(args.patient_data_path, sheet_name = 1) # Read the second sheet - oct_pacs_data_modified+VA

    # Preprocess the data
    label_df, unique_patient_ids = preprocess_label_data(label_df)
    patient_data_df = preprocess_patient_data(patient_data_df, unique_patient_ids)

    print(patient_data_df.head())

if __name__ == '__main__':
    main()