from os import path
import pandas as pd
from globals import RAW_PRISM_CSV, DATA_DIR, INDEX, FROM_PRISM, DATASET_ANALYSIS_COL_MAP
from globals import SOC_ISO_FEATS_RAW, PSYCH_DIST_FEATS
from globals import TRUE_COL_MAP, BLANK_COL_MAP

TUMOUR_SITE_MAP = {
    'Breast': 'Breast',
    'Colon/Rectum': 'Gastro-Intestinal',
    'Pancreas': 'Gastro-Intestinal',
    'Esophagus (6th - All,  7th - Adeno)': 'Gastro-Intestinal',
    'Esophagus (7th - Squam and non-Adeno)': 'Gastro-Intestinal',
    'Anus / Anal Canal': 'Gastro-Intestinal',
    'Stomach': 'Gastro-Intestinal',
    'Liver': 'Gastro-Intestinal',
    'Gallbladder': 'Gastro-Intestinal',
    'Appendix - Carcinoma (7th Ed)': 'Gastro-Intestinal',
    'Small Intestine': 'Gastro-Intestinal',
    'Ampulla of Vater': 'Gastro-Intestinal',
    'Distal Bile Duct (7th Ed)': 'Gastro-Intestinal',
    'Perihilar Bile Ducts (7th Ed)': 'Gastro-Intestinal',
    'Intrahepatic Bile Ducts (7th Ed)': 'Gastro-Intestinal',
    'Bile Ducts (6th Ed)': 'Gastro-Intestinal',
    'NET - Small Intestine (7th Ed)': 'Gastro-Intestinal',
    'NET - Colon and Rectum (7th Ed)': 'Gastro-Intestinal',
    'GIST - Gastric and Omental (7th Ed)': 'Gastro-Intestinal',
    'GIST - Non-Gastric (7th Ed)': 'Gastro-Intestinal',
    'Appendix - Carcinoid (7th Ed)': 'Gastro-Intestinal',
    'Lung': 'Thoracic',
    'Mesothelioma': 'Thoracic',
    'Prostate': 'Prostate',
    'Corpus Uteri - Carcinoma': 'Gynaecologic',
    'Ovary': 'Gynaecologic',
    'Cervix Uteri': 'Gynaecologic',
    'Vulva': 'Gynaecologic',
    'Vagina': 'Gynaecologic',
    'Fallopian Tube': 'Gynaecologic',
    'Gestational Trophoblastic': 'Gynaecologic',
    'Corpus Uteri - Leio/Stromal Sarcoma (7th Ed)': 'Gynaecologic',
    'Corpus Uteri - Adenosarcoma (7th Ed)': 'Gynaecologic',
    'Skin Carcinoma': 'Skin',
    'Melanoma of the Skin (7th Ed)': 'Skin',
    'Merkel Cell Carcinoma (7th Ed)': 'Skin',
    'Eyelid Carcinoma': 'Skin',
    'Conjunctiva - Melanoma': 'Skin',
    'Melanoma - Iris': 'Skin',
    'Melanoma - Ciliary Body / Choroid': 'Skin',
    'Mucosal Melanoma - Head / Neck (7th Ed)': 'Skin',
    'Hodgkin / Non-Hodgkin Lymphoma (7th Ed)': 'Haematologic',
    'Multiple Myeloma / Plasma Cell (7th Ed)': 'Haematologic',
    'Ocular Adnexal Lymphoma (7th Ed)': 'Haematologic',
    'Primary Cutaneous Lymphomas (7th Ed)': 'Haematologic',
    'Pediatric Lymphoid Malignancy (7th Ed)': 'Haematologic',
    'Oropharynx': 'Head and Neck',
    'Lip / Oral Cavity': 'Head and Neck',
    'Nasopharynx': 'Head and Neck',
    'Major Salivary Glands': 'Head and Neck',
    'Larynx - Glottis': 'Head and Neck',
    'Larynx - Supraglottis': 'Head and Neck',
    'Hypopharynx': 'Head and Neck',
    'Nasal Cavity / Ethmoids': 'Head and Neck',
    'Maxillary Sinus': 'Head and Neck',
    'Larynx - Subglottis': 'Head and Neck',
}

TUMOUR_SITE_ORDER = [
    'Breast',
    'Gastro-Intestinal',
    'Thoracic',
    'Prostate',
    'Gynaecologic',
    'Skin',
    'Haematologic',
    'Head and Neck',
    'Other',
]


def append_row(rows, section, characteristic, n=None, percent=None, missing_n=None, mean=None, sd=None, median=None):
    rows.append({
        'section': section,
        'characteristic': characteristic,
        'n': n,
        '%': percent,
        'missing_n': missing_n,
        'mean': mean,
        'sd': sd,
        'median': median,
    })


def percent_non_missing(count, total_n, missing_n):
    if missing_n in [None, 'N/A']:
        denominator = total_n
    else:
        denominator = total_n - missing_n
    if denominator == 0:
        return None
    return round(count * 100 / denominator, 1)


def build_tumour_site_counts(raw_df):
    tumour = raw_df['MajorGroupsStagingDesc'].fillna('').astype(str).str.strip()
    mapped = tumour.map(lambda value: TUMOUR_SITE_MAP.get(value, 'Other'))
    return mapped.value_counts().reindex(TUMOUR_SITE_ORDER, fill_value=0)


def build_composite_counts(raw_df):
    # Any domain of social isolation based on the five individual raw domains, not Social_isolation_YN.
    social_cols = ['LiveAlone', 'LostSpousePartner', 'Assistance', 'RegularContact', 'EmotionalSupport']
    social_df = raw_df[social_cols].astype(str).apply(lambda s: s.str.strip())
    positive_social = pd.DataFrame({
        'LiveAlone': social_df['LiveAlone'].isin(['Y', '1', '1.0']),
        'LostSpousePartner': social_df['LostSpousePartner'].isin(['Y', '1', '1.0']),
        'Assistance': social_df['Assistance'].isin(['N', '0', '0.0']),
        'RegularContact': social_df['RegularContact'].isin(['N', '0', '0.0']),
        'EmotionalSupport': social_df['EmotionalSupport'].isin(['N', '0', '0.0']),
    })
    blank_social = social_df.isin(['B', '99', '99.0', 'blank!', 'nan'])
    any_social = positive_social.any(axis=1).sum()
    social_missing = blank_social.all(axis=1).sum()

    psych_df = raw_df[PSYCH_DIST_FEATS].astype(str).apply(lambda s: s.str.strip())
    positive_psych = psych_df.isin(['2', '2.0', '2.00']).any(axis=1).sum()
    psych_missing = psych_df.isin(['0', '0.0', '0.00', 'blank!', 'nan']).all(axis=1).sum()

    return {
        'any_social_iso_n': int(any_social),
        'any_social_iso_missing_n': int(social_missing),
        'any_psych_distress_n': int(positive_psych),
        'any_psych_distress_missing_n': int(psych_missing),
    }


def build_patient_characteristics_table(pts_csv=RAW_PRISM_CSV, data_dir=DATA_DIR):
    output_path = f'{data_dir}patient_characteristics_table.csv'
    raw_df = pd.read_csv(pts_csv, index_col=INDEX, engine='python', encoding='latin1')
    n_pts = len(raw_df)
    rows = []

    append_row(
        rows,
        'Continuous variables',
        'Age at diagnosis (years)',
        mean=round(raw_df['AgeAtDiagnosis'].mean(), 1),
        sd=round(raw_df['AgeAtDiagnosis'].std(), 1),
        median=round(raw_df['AgeAtDiagnosis'].median(), 1),
    )
    append_row(
        rows,
        'Continuous variables',
        'Length of follow-up (months until death or last follow-up)',
        mean=round(raw_df['Dx_to_death_lastfu_months'].mean(), 1),
        sd=round(raw_df['Dx_to_death_lastfu_months'].std(), 1),
        median=round(raw_df['Dx_to_death_lastfu_months'].median(), 1),
    )

    female_n = int((raw_df['Sex'] == 'F').sum())
    male_n = int((raw_df['Sex'] == 'M').sum())
    alive_n = int((raw_df['PatStatus'] == 'A').sum())

    append_row(rows, 'Patient characteristics', 'Female', female_n, percent_non_missing(female_n, n_pts, 0), 0)
    append_row(rows, 'Patient characteristics', 'Male', male_n, percent_non_missing(male_n, n_pts, 0), 0)
    append_row(rows, 'Patient characteristics', 'Alive at last follow-up', alive_n, percent_non_missing(alive_n, n_pts, 0), 0)

    tumour_counts = build_tumour_site_counts(raw_df)
    for category in TUMOUR_SITE_ORDER:
        count = int(tumour_counts[category])
        append_row(rows, 'Tumour site', category, count, percent_non_missing(count, n_pts, 'N/A'), 'N/A')

    metastatic_counts = raw_df['M1_IV_ALL_CAT'].astype(str).str.strip().value_counts()
    metastatic_n = int(metastatic_counts.get('1', 0))
    non_metastatic_n = int(metastatic_counts.get('0', 0))
    unknown_metastatic_n = int(metastatic_counts.get('99', 0))
    append_row(rows, 'Metastatic status at diagnosis', 'Metastatic', metastatic_n, percent_non_missing(metastatic_n, n_pts, unknown_metastatic_n), unknown_metastatic_n)
    append_row(rows, 'Metastatic status at diagnosis', 'Non-metastatic', non_metastatic_n, percent_non_missing(non_metastatic_n, n_pts, unknown_metastatic_n), unknown_metastatic_n)
    append_row(rows, 'Metastatic status at diagnosis', 'Unknown', unknown_metastatic_n, round(unknown_metastatic_n * 100 / n_pts, 2), unknown_metastatic_n)

    social_specs = [
        ('Lives Alone', 'LiveAlone'),
        ('Lost Partner Recently', 'LostSpousePartner'),
        ('No Regular Contact', 'RegularContact'),
        ('No Emotional Support', 'EmotionalSupport'),
        ('No IADL Assistance', 'Assistance'),
    ]
    for label, col in social_specs:
        raw_col = raw_df[col].astype(str).str.strip()
        if col in ['Assistance', 'RegularContact', 'EmotionalSupport']:
            positive_n = int(raw_col.isin(['N', '0', '0.0']).sum())
        else:
            positive_n = int(raw_col.isin(['Y', '1', '1.0']).sum())
        missing_n = int(raw_col.isin(['B', '99', '99.0', 'blank!', 'nan']).sum())
        append_row(rows, 'Domains of social isolation', label, positive_n, percent_non_missing(positive_n, n_pts, missing_n), missing_n)

    composite_counts = build_composite_counts(raw_df)
    append_row(
        rows,
        'Domains of social isolation',
        'Any Domain of Social Isolation',
        composite_counts['any_social_iso_n'],
        percent_non_missing(composite_counts['any_social_iso_n'], n_pts, composite_counts['any_social_iso_missing_n']),
        composite_counts['any_social_iso_missing_n'],
    )

    depression_raw = raw_df['DepressionScore_CAT2'].astype(str).str.strip()
    anxiety_raw = raw_df['AnxietyScore_CAT2'].astype(str).str.strip()
    depression_n = int(depression_raw.isin(['2', '2.0', '2.00']).sum())
    anxiety_n = int(anxiety_raw.isin(['2', '2.0', '2.00']).sum())
    append_row(rows, 'Psychological distress', 'Depression Symptoms', depression_n, percent_non_missing(depression_n, n_pts, 0), 0)
    append_row(rows, 'Psychological distress', 'Anxiety Symptoms', anxiety_n, percent_non_missing(anxiety_n, n_pts, 0), 0)
    append_row(
        rows,
        'Psychological distress',
        'Any Psychological Distress',
        composite_counts['any_psych_distress_n'],
        percent_non_missing(composite_counts['any_psych_distress_n'], n_pts, 0),
        0,
    )

    paper_df = pd.DataFrame(rows)
    paper_df.to_csv(output_path, index=False)
    print(f"patient characteristics table written to {output_path}")
    return paper_df


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
    build_patient_characteristics_table()
    print("Analysis complete!")
