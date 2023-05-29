import sys
import argparse
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", type=Path)
    parser.add_argument("--patient_data_path", type=Path)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--valid_ratio", type=float, default=0.2)

    parser.add_argument("--output_dir", type=Path, default=Path("./data_keras"))
    parser.add_argument("--original_img_dir", type=Path, default=Path("./data_split"))

    args = parser.parse_args()
    return args

def split_os_od(df):
    df_os = df[df['Filename'].str.contains('OS')]
    df_od = df[df['Filename'].str.contains('OD')]
    return df_os, df_od

def copy_image(df, origin_dir, destin_dir):
    for _, row in df.iterrows():
        image_path = os.path.join(origin_dir, row['Patient ID'], row['Filename'])
        destination_path = os.path.join(destin_dir, row['Filename'])
        shutil.copy(image_path, destination_path)

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

def duplicated_rows(df):
    df_top_img = df
    df_down_img = df.copy()

    # rename filename
    df_top_img['Filename'] = df_top_img['Filename'].apply(lambda x: x.replace('.jpg', '_top.jpg'))
    df_down_img['Filename'] = df_down_img['Filename'].apply(lambda x: x.replace('.jpg', '_down.jpg'))

    df_duplicated = pd.concat([df_top_img, df_down_img], ignore_index=True)

    
    return df_duplicated

def do_sanity_check(df):
    expected_values = [0, 1]
    for col in df.columns:
        if col == "Patient ID" or col == "Filename":
            continue
        elif col == "HCQ_label":
            # only contain 0 or 1 in label column
            assert df[col].isin(expected_values).all()
        else:
            pass

def replace_special_value_in_label_col(df, col_name):
    # Define the pattern and replacement values
    pattern = r'^([0-1X])-.*'
    replacement = {
        '0': 0, 
        '1': 1, 
        # 'X': 1
        }

    # Apply the regex pattern and replacement using the replace() method
    df[col_name] = df[col_name].str.replace(
        pattern, 
        lambda x: str(replacement.get(x.group(1), x.group(1))), 
        regex=True
        )
    
    # df[col_name] = df[col_name].replace('X', 1)
    df[col_name] = df[col_name].fillna(-1).astype(int)
    return df

def fill_detailed_region(df):
    # Fill columns 
    # EZ-fovea, EZ-parafovea, EZ-perifovea, EZ-pericentral 
    # RPE-fovea, RPE-parafovea, RPE-perifovea, RPE-pericentral to 0
    columns_to_fill = ['EZ-fovea', 'EZ-parafovea', 'EZ-perifovea', 'EZ-pericentral',
                       'RPE-fovea', 'RPE-parafovea', 'RPE-perifovea', 'RPE-pericentral']
    df[columns_to_fill] = df[columns_to_fill].fillna(0)

    return df

def remove_od_os_with_X(df):
    df = df[
        (~df['od'].astype(str).str.contains('X')) & \
            (~df['os'].astype(str).str.contains('X'))
        ]
    return df

def mix_os_od_column(df):
    df['HCQ_label'] = df['od'].fillna(df['os'])
    return df

def preprocess_patient_data(df, patient_ids):
    # get only OCT data
    df = df[df["Exam type"] == "OCT"]

    # filter with unique patient ids
    df = df[df["Patient ID"].isin(patient_ids)]

    # keep od, os, EZ-fovea, EZ-parafovea, EZ-perifovea, EZ-pericentral, RPE-fovea, RPE-parafovea, RPE-perifovea, RPE-pericentral columns
    df = remove_and_rename_columns(df)

    # remove row that contains X
    df = remove_od_os_with_X(df)

    # od, os, : 0-XYZ -> 0, 1-XYZ -> 1, nan -> -1
    df = replace_special_value_in_label_col(df, "od")
    df = replace_special_value_in_label_col(df, "os")

    # mix os and od column -> HCQ_labels
    df = mix_os_od_column(df)

    # remove HQC labels that are -1 (means od and os == 0)
    df = df[df["HCQ_label"] != -1]

    # fill EZ-fovea, EZ-parafovea, EZ-perifovea, EZ-pericentral, 
    # RPE-fovea, RPE-parafovea, RPE-perifovea, RPE-pericentral
    # to 0
    df = fill_detailed_region(df)

    # duplicated rows, since each picture has top and bottom pictures
    df = duplicated_rows(df)

    # Sanity check
    do_sanity_check(df)

    return df 

def preprocess_label_data(label_df):
    patient_with_labels = label_df[label_df["label"] == 1]
    unique_patient_ids = patient_with_labels["Patient_ID"].unique()
    return patient_with_labels, unique_patient_ids

def split_data(data, train_ratio, valid_ratio, test_ratio, random_state=42):
    assert train_ratio + valid_ratio + test_ratio == 1.0, "Ratios should sum up to 1.0"
    
    # Split the data into train and remaining sets
    remaining_ratio = valid_ratio + test_ratio
    train_data, remaining_data = train_test_split(
        data, test_size=remaining_ratio, 
        random_state=random_state
        )
    
    # Calculate the remaining ratio as a fraction of the total remaining ratio
    relative_frac_test = test_ratio / (valid_ratio + test_ratio)
    
    # Split the remaining data into validation and test sets
    valid_data, test_data = train_test_split(
        remaining_data, test_size = relative_frac_test, 
        random_state=random_state
        )
    
    return train_data, valid_data, test_data

def main():
    args = parse_args()

    # Read the data
    label_df = pd.read_excel(args.label_path, engine='openpyxl')
    patient_data_df = pd.read_excel(args.patient_data_path, sheet_name = 1, engine='openpyxl') # Read the second sheet - oct_pacs_data_modified+VA

    # Preprocess the data
    label_df, unique_patient_ids = preprocess_label_data(label_df)
    patient_data_df = preprocess_patient_data(patient_data_df, unique_patient_ids)

    # print(patient_data_df.head())
    # exit()
    # print(patient_data_df[patient_data_df["Patient ID"] == "P215350000002"])

    # cut to train valid test for keras package 
    train_ds, valid_ds, test_ds = \
        split_data(
        patient_data_df, 
        args.train_ratio, 
        args.valid_ratio, 
        1-args.train_ratio-args.valid_ratio
        )

    train_ds.to_csv(
        os.path.join(args.output_dir, "train_os.csv"), 
        index=False
        )
    valid_ds.to_csv(
        os.path.join(args.output_dir, "val_os.csv"), 
        index=False
        )
    test_ds.to_csv(
        os.path.join(args.output_dir, "test_os.csv"), 
        index=False
        )
    
    # Move files to the correct directory
    train_path = os.path.join(args.output_dir, "train")
    valid_path = os.path.join(args.output_dir, "val")
    test_path = os.path.join(args.output_dir, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(valid_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    copy_image(train_ds, args.original_img_dir, train_path)
    copy_image(valid_ds, args.original_img_dir, valid_path)
    copy_image(test_ds, args.original_img_dir, test_path)
    
if __name__ == '__main__':
    main()