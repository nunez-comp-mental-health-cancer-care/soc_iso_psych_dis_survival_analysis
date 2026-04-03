import pandas as pd
import numpy as np
import warnings
# from data_globals import FROM_CANCER_DETAILS, INDEX, LATEST_DATE, FROM_PSSCAN_DETAILS, DISCIPLINES_OF_INTEREST, DISCIPLINE
# from data_globals import PSSCAN_EMOTS, PSSCAN_INFOS, PSSCAN_PRACS, PSSCAN_PHYIS, PSSCAN_SPIRS, PSSCAN_SOCLS, PSSCAN_ALL
# from data_globals import FROM_DOC_DETAILS, DATE_FEATURES
# from data_globals import RAW_CANCER_CSV, RAW_PSSCAN_CSV, RAW_DOC_CSV, DATA_DIR
from utils import dif_in_months, add_cols, parse_multi_to_datetime
from dateutil.relativedelta import relativedelta
from os import path
from datetime import timedelta
from globals import RAW_PRISM_CSV, DATA_DIR, INDEX, FROM_PRISM, PSYCH_DIST_FEATS
from globals import SOC_ISO_FEATS_RAW, SOC_ISO_FEATS_TO_INVERT


def coerce_to_bool(series, true_values, false_values):
    """
    Normalize mixed PRISM encodings to a strict boolean series.
    """
    normalized = series.astype(str).str.strip()
    true_values = {str(value).strip() for value in true_values}
    false_values = {str(value).strip() for value in false_values}

    mapped = normalized.map(lambda value: True if value in true_values else False if value in false_values else pd.NA)
    return mapped.fillna(False).astype(bool)


def df_analysis(df, filename):
    """
    Function to analysize

    :param df:
    :param filename:
    :return:
    """


def preprocess(pts_csv=RAW_PRISM_CSV, data_dir=DATA_DIR, overwrite=False):
    filename = f'{data_dir}preprocessed_pts.csv'

    # Check if base preprocessed pts file already generated, return if so unless overwrite=True
    if path.exists(filename) and not (overwrite):
        print("preprocessed_pts.csv already generated, will load and exit preprocesser")
        return pd.read_csv(filename, index_col=INDEX)

    # Otherwise, generate
    print("preprocessed_pts.csv is processing, this may take a few minutes")
    pts_df = pd.read_csv(pts_csv, index_col=INDEX, engine='python', encoding='latin1')
    df = pts_df.copy()[FROM_PRISM]  # Pull in only needed columns

    # Months observed use "Dx_to_death_lastfu_months" instead of calculating ourselves
    # df['obs_mo'] = df.apply(lambda row: dif_in_months(row, 'DiagnosisDate', 'Death_or_lastfu_date'), axis=1)
    # df = df.drop(columns=['DiagnosisDate', 'Death_or_lastfu_date'])

    # Fill blank Alive or Dead, Convert to True and False, replace blanks with False
    # print(df[1250:1260]['PatStatus'])
    # df['PatStatus'] = df['PatStatus'].replace(r'^\s*$', False, regex=True)
    df['PatStatus'] = df['PatStatus'].fillna(False)
    df['PatStatus'] = coerce_to_bool(df['PatStatus'], true_values=['D'], false_values=['A', 'False', 'blank!', 'nan'])

    # Replace missing metastatic entires with False, and convert to True False
    df['M1_IV_ALL_CAT'] = coerce_to_bool(df['M1_IV_ALL_CAT'], true_values=['1', '1.0'], false_values=['0', '0.0', '99', '99.0', 'blank!', 'nan'])

    # For the various social isolation features, fill blanks with mode, and convert to True/False
    for feature in SOC_ISO_FEATS_RAW:
        df[feature] = df[feature].astype('str')  # Convert col to string
        mode = df[feature].mode().iat[0]  # Find mode of column
        # Replace Blanks, coded as B or 99, with mode of the column, and convert Y N, 1 0 to True False
        df[feature] = df[feature].replace({"B": mode, "99": mode})
        df[feature] = coerce_to_bool(df[feature], true_values=['Y', '1', '1.0'], false_values=['N', '0', '0.0'])

    for feature in PSYCH_DIST_FEATS:
        df[feature] = coerce_to_bool(df[feature], true_values=['2', '2.0'], false_values=['1', '1.0', '0', '0.0', 'nan'])

    # Some of the social isolation features need to be flipped so that True = has an expected bad thing
    for feature in SOC_ISO_FEATS_TO_INVERT:
        new_feature_name = 'No' + feature
        df[new_feature_name] = ~df[feature]
        df = df.drop(feature, axis=1)

    # Convert sex
    df['Sex'] = coerce_to_bool(df['Sex'], true_values=['M'], false_values=['F'])

    # Rename columns
    df = df.rename(columns={"PatStatus": "died", "Dx_to_death_lastfu_months": "observed_mo", 'Sex': 'male',
                            'AgeAtDiagnosis': 'age_dx'})

    # print(df.dtypes) # check data dtypes
    df.to_csv(filename)

    print('preprocessing complete!')
    return df


def index_ends_10(row):
    """
    Quick helper function to make a random dspln column based on whether pt's index ends with 10.
    :param row: row from df
    :return: 1 if index ends with 10, 0 if not
    """
    if row.name % 10 == 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    preprocess(overwrite=True)
    # preprocess_pts(psyhx_subset='anx_or_dep')
    # preprocess_pts(help_subset='alone_or_no_assist')
    # preprocessed_df = preprocess_pts(cancer_csv, psscan_csv, doc_csv, data_dir, dev_mode=True)
    # preprocessed_docs = preprocess_docs()
    print("Pre-processing complete!")
