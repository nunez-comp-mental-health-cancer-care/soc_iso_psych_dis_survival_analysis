import pandas as pd
from os import path
from globals import RAW_PRISM_CSV, DATA_DIR, INDEX, FROM_PRISM, DATASET_ANALYSIS_COL_MAP
from globals import SOC_ISO_FEATS_RAW, SOC_ISO_FEATS_TO_INVERT


def analyze_dataset(pts_csv=RAW_PRISM_CSV, data_dir=DATA_DIR, overwrite=False):


    filename = f'{data_dir}dataset_analysis.csv'

    # Check if base preprocessed pts file already generated, return if so unless overwrite=True
    if path.exists(filename) and not(overwrite):
        print("dataset_analysis.csv already generated, will load and exit preprocesser")
        return pd.read_csv(filename, index_col=INDEX)

    # Otherwise, generate
    print("dataset_analysis.csv is being generated, this may take a few minutes")
    pts_df = pd.read_csv(pts_csv, index_col=INDEX, engine='python')
    df = pts_df.copy()[FROM_PRISM] # Pull in only needed columns

    # Prepare df
    metrics = ['%', 'n', 'blank', 'median', 'mean', 'sd']
    analysis_df = pd.DataFrame(columns=metrics, index=FROM_PRISM)

    # Rename columns
    analysis_df = analysis_df.rename(index=DATASET_ANALYSIS_COL_MAP)

    # Alive or Dead
    #print(df[df['PatStatus'] == 'A'].count())
    print(df['PatStatus'].squeeze().value_counts())
    #analysis_df.loc['Alive at end of fu', 'n'] = df[df['PatStatus'] == 'A'].count()

    print(analysis_df)


if __name__ == '__main__':
    analyze_dataset(overwrite=True)
    print("Analysis complete!")