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


def normalize_raw_value(value):
    """
    Normalize mixed raw CSV values for comparison and QC.
    """
    if pd.isna(value):
        return 'nan'
    return str(value).strip()


def coerce_to_bool(series, true_values, false_values, missing_values=None, col_name=None):
    """
    Normalize mixed PRISM encodings to a nullable boolean series and capture QC stats.
    """
    missing_values = missing_values or []
    normalized = series.map(normalize_raw_value)
    true_values = {normalize_raw_value(value) for value in true_values}
    false_values = {normalize_raw_value(value) for value in false_values}
    missing_values = {normalize_raw_value(value) for value in missing_values}

    mapped = normalized.map(
        lambda value: True if value in true_values
        else False if value in false_values
        else pd.NA if value in missing_values
        else pd.NA
    )

    qc_rows = []
    value_counts = normalized.value_counts(dropna=False).sort_index()
    for raw_value, count in value_counts.items():
        if raw_value in true_values:
            mapped_class = 'true'
        elif raw_value in false_values:
            mapped_class = 'false'
        elif raw_value in missing_values:
            mapped_class = 'missing'
        else:
            mapped_class = 'unmapped'

        qc_rows.append({
            'column': col_name,
            'raw_value': raw_value,
            'count': int(count),
            'mapped_class': mapped_class,
        })

    return mapped.astype('boolean'), qc_rows


def append_qc_rows(qc_rows, col_name, raw_series, category_map):
    """
    Capture QC counts for the original raw values and how they are handled.
    """
    value_counts = raw_series.map(normalize_raw_value).value_counts(dropna=False).sort_index()
    for raw_value, count in value_counts.items():
        qc_rows.append({
            'column': col_name,
            'raw_value': raw_value,
            'count': int(count),
            'mapped_class': category_map.get(raw_value, 'unmapped'),
        })


def qc_summary_df(qc_rows):
    qc_df = pd.DataFrame(qc_rows)
    if qc_df.empty:
        return qc_df, qc_df

    summary_df = (
        qc_df.pivot_table(
            index='column',
            columns='mapped_class',
            values='count',
            aggfunc='sum',
            fill_value=0,
        )
        .reset_index()
    )

    for col in ['true', 'false', 'missing', 'imputed', 'unmapped']:
        if col not in summary_df.columns:
            summary_df[col] = 0

    summary_df = summary_df[['column', 'true', 'false', 'missing', 'imputed', 'unmapped']]
    return qc_df, summary_df


def df_analysis(df, filename):
    """
    Function to analysize

    :param df:
    :param filename:
    :return:
    """


def preprocess(pts_csv=RAW_PRISM_CSV, data_dir=DATA_DIR, overwrite=False, strict=True):
    filename = f'{data_dir}preprocessed_pts.csv'
    qc_filename = f'{data_dir}preprocess_qc.csv'
    qc_summary_filename = f'{data_dir}preprocess_qc_summary.csv'

    # Check if base preprocessed pts file already generated, return if so unless overwrite=True
    if path.exists(filename) and not (overwrite):
        print("preprocessed_pts.csv already generated, will load and exit preprocesser")
        return pd.read_csv(filename, index_col=INDEX)

    # Otherwise, generate
    print("preprocessed_pts.csv is processing, this may take a few minutes")
    pts_df = pd.read_csv(pts_csv, index_col=INDEX, engine='python', encoding='latin1')
    df = pts_df.copy()[FROM_PRISM]  # Pull in only needed columns
    qc_rows = []

    # Months observed use "Dx_to_death_lastfu_months" instead of calculating ourselves
    # df['obs_mo'] = df.apply(lambda row: dif_in_months(row, 'DiagnosisDate', 'Death_or_lastfu_date'), axis=1)
    # df = df.drop(columns=['DiagnosisDate', 'Death_or_lastfu_date'])

    # Fill blank Alive or Dead, Convert to True and False, replace blanks with False
    # print(df[1250:1260]['PatStatus'])
    # df['PatStatus'] = df['PatStatus'].replace(r'^\s*$', False, regex=True)
    raw_patstatus = df['PatStatus'].copy()
    append_qc_rows(qc_rows, 'PatStatus', raw_patstatus, {
        'A': 'false',
        'D': 'true',
        'False': 'imputed',
        'blank!': 'imputed',
        'nan': 'imputed',
    })
    df['PatStatus'] = df['PatStatus'].fillna(False)
    df['PatStatus'] = df['PatStatus'].replace({"blank!": False, 'A': False, 'D': True})

    # Replace missing metastatic entires with False, and convert to True False
    raw_metastatic = df['M1_IV_ALL_CAT'].copy()
    append_qc_rows(qc_rows, 'M1_IV_ALL_CAT', raw_metastatic, {
        '0': 'false',
        '0.0': 'false',
        '1': 'true',
        '1.0': 'true',
        '99': 'imputed',
        '99.0': 'imputed',
        'blank!': 'imputed',
        'nan': 'imputed',
    })
    df['M1_IV_ALL_CAT'] = df['M1_IV_ALL_CAT'].astype('str')
    df['M1_IV_ALL_CAT'] = df['M1_IV_ALL_CAT'].replace({"blank!": False, '0': False, '1': True, '99': False})

    # For the various social isolation features, fill blanks with mode, and convert to True/False
    for feature in SOC_ISO_FEATS_RAW:
        raw_feature = df[feature].copy()
        normalized_raw = raw_feature.map(normalize_raw_value)
        non_blank_values = normalized_raw.loc[~normalized_raw.isin(['B', '99', '99.0', 'blank!', 'nan'])]
        mode = non_blank_values.mode().iat[0]
        mode_class = 'true' if mode in {'Y', '1', '1.0'} else 'false'
        append_qc_rows(qc_rows, feature, raw_feature, {
            'Y': 'true',
            '1': 'true',
            '1.0': 'true',
            'N': 'false',
            '0': 'false',
            '0.0': 'false',
            'B': 'imputed',
            '99': 'imputed',
            '99.0': 'imputed',
            'blank!': 'imputed',
            'nan': 'imputed',
        })
        df[feature] = df[feature].astype('str')
        replacement_value = {'true': True, 'false': False}[mode_class]
        df[feature] = df[feature].replace({"B": mode, "99": mode, "99.0": mode, "blank!": mode, "nan": mode})
        df[feature] = df[feature].replace({"Y": True, "N": False, "1": True, "0": False, "1.0": True, "0.0": False})
        df[feature] = df[feature].replace({mode: replacement_value}).astype(bool)

    for feature in PSYCH_DIST_FEATS:
        raw_feature = df[feature].copy()
        append_qc_rows(qc_rows, feature, raw_feature, {
            '2': 'true',
            '2.0': 'true',
            '1': 'false',
            '1.0': 'false',
            '0': 'imputed',
            '0.0': 'imputed',
            'blank!': 'imputed',
            'nan': 'imputed',
        })
        df[feature] = df[feature].astype('str')
        df[feature] = df[feature].replace({"2.0": True, "1.0": False, "0.0": False, "2": True, "1": False, "0": False}).astype(bool)

    # Some of the social isolation features need to be flipped so that True = has an expected bad thing
    for feature in SOC_ISO_FEATS_TO_INVERT:
        new_feature_name = 'No' + feature
        df[new_feature_name] = ~df[feature]
        df = df.drop(feature, axis=1)

    # Convert sex
    raw_sex = df['Sex'].copy()
    append_qc_rows(qc_rows, 'Sex', raw_sex, {
        'M': 'true',
        'F': 'false',
    })
    df['Sex'] = df['Sex'].replace({'M': True, 'F': False})

    qc_df, qc_summary = qc_summary_df(qc_rows)
    qc_df.to_csv(qc_filename, index=False)
    qc_summary.to_csv(qc_summary_filename, index=False)

    unmapped_df = qc_df.loc[qc_df['mapped_class'] == 'unmapped'].copy()
    if not unmapped_df.empty:
        print(f"Found unmapped values. Detailed QC written to {qc_filename}")
        print(unmapped_df.to_string(index=False))
        if strict:
            raise ValueError("Unmapped raw values detected during preprocessing. Review preprocess_qc.csv and update mappings.")
        warnings.warn("Unmapped raw values detected during preprocessing; leaving them as missing values.", stacklevel=2)

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
