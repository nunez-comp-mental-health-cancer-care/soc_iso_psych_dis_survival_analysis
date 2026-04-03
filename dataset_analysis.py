from os import path
import pandas as pd
from globals import RAW_PRISM_CSV, DATA_DIR, INDEX, FROM_PRISM, DATASET_ANALYSIS_COL_MAP
from globals import SOC_ISO_FEATS_RAW, PSYCH_DIST_FEATS
from globals import TRUE_COL_MAP, BLANK_COL_MAP


def analyze_dataset(pts_csv=RAW_PRISM_CSV, data_dir=DATA_DIR, overwrite=False):
    filename = f'{data_dir}dataset_analysis.csv'

    # Check if base preprocessed pts file already generated, return if so unless overwrite=True
    if path.exists(filename) and not overwrite:
        print("dataset_analysis.csv already generated, will load and exit preprocesser")
        return pd.read_csv(filename, index_col=INDEX)

    # Otherwise, generate
    print("dataset_analysis.csv is being generated, this may take a few minutes")
    pts_df = pd.read_csv(pts_csv, index_col=INDEX, engine='python', encoding='latin1')
    df = pts_df.copy()[FROM_PRISM]  # Pull in only needed columns

    # Prepare df
    metrics = ['%', 'n', 'blank', 'median', 'mean', 'sd']
    analysis_df = pd.DataFrame(columns=metrics, index=FROM_PRISM)

    # Count number of patients
    n_pts = len(df.index)

    # Alive or Dead
    # print(df[df['PatStatus'] == 'A'].count())
    # patstatus_vals = df['PatStatus'].squeeze().value_counts()
    # analysis_df.loc['PatStatus', 'n'] = patstatus_vals['A']
    # analysis_df.loc['PatStatus', 'blank'] = n_pts - patstatus_vals['A'] - patstatus_vals['D']

    # Binary Columns that are 0,1, or #Null!
    binary_cols = SOC_ISO_FEATS_RAW + PSYCH_DIST_FEATS + ['M1_IV_ALL_CAT', 'Sex', 'PatStatus']
    for col in binary_cols:
        # Find values and counts for each value of a col
        col_vals = df[col].squeeze().value_counts()

        # Differnt columns have different encodings, find what is True and Blank for the col
        col_true_val = TRUE_COL_MAP[col]
        col_blank_val = BLANK_COL_MAP[col]

        # Calculate the number of True and Blanks
        analysis_df.loc[col, 'n'] = col_vals[col_true_val]
        try:
            analysis_df.loc[col, 'blank'] = col_vals[col_blank_val]
        except:
            analysis_df.loc[col, 'blank'] = 0  # Some columns have no blanks

    # Calculate % for the binary columns
    analysis_df['%'] = analysis_df['n'] * 100 / n_pts

    # Calculate median and sd for the int columns
    int_cols = ['Dx_to_death_lastfu_months', 'AgeAtDiagnosis']
    for col in int_cols:
        analysis_df.loc[col, 'median'] = df[col].squeeze().median()
        print(df[col].median())
        analysis_df.loc[col, 'mean'] = df[col].squeeze().mean()
        analysis_df.loc[col, 'sd'] = df[col].squeeze().std()

    # Rename and round columns
    analysis_df = analysis_df.rename(index=DATASET_ANALYSIS_COL_MAP)
    analysis_df = analysis_df.astype(float).round(2)

    analysis_df.to_csv(filename)


if __name__ == '__main__':
    analyze_dataset(overwrite=True)
    print("Analysis complete!")
