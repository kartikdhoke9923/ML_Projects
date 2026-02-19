import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

def run_data_cleaning():

    # --------- Path Setup ---------
    base_dir = os.getcwd()

    input_path = os.path.join(base_dir, "notebook", "data", "application_train.csv")
    output_path = os.path.join(base_dir, "notebook", "data", "cleaned.csv")

    # --------- Read Raw Data ---------
    df = pd.read_csv(input_path)

    # --------- Drop >60% Missing Columns ---------
    means = df.isna().mean() * 100
    c = [x for x, y in means.items() if y > 60]
    df = df.drop(columns=c)

    # --------- Fix Dirty Employment Value ---------
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    df['DAYS_EMPLOYED_MISSING'] = df['DAYS_EMPLOYED'].isna().astype(int)
    df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365

    df.drop(["DAYS_EMPLOYED"], axis=1, inplace=True)
    df.drop(["DAYS_EMPLOYED_MISSING"], axis=1, inplace=True)

    # --------- Drop ID Column ---------
    df.drop(columns=['SK_ID_CURR'], inplace=True, errors='ignore')

    # --------- Create Age Feature ---------
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df.drop(["DAYS_BIRTH"], axis=1, inplace=True)

    # --------- Drop Imbalanced Binary Columns ---------
    binary_columns = []
    drop_binaries = []

    for col in df.columns:
        value_percentages = df[col].value_counts(normalize=True) * 100
        if len(value_percentages) == 2:
            binary_columns.append(col)

    if "TARGET" in binary_columns:
        binary_columns.remove("TARGET")

    for col in binary_columns:
        dist = df[col].value_counts(normalize=True)
        minority_pct = dist.min() * 100
        if minority_pct < 10:
            drop_binaries.append(col)

    df.drop(columns=drop_binaries, inplace=True)

    # --------- Remove PROCESS_START Columns ---------
    ps = [x for x in df.columns if "PROCESS_START" in x]
    df = df.drop(columns=ps)

    # --------- Save Cleaned Data ---------
    df.to_csv(output_path, index=False)

    print("Data Cleaning Completed ✔")
