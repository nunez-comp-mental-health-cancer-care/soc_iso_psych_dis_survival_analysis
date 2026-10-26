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
    'NET - Gastric (7th Ed)': 'Gastro-Intestinal',
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
    'Urethra': 'Genitourinary (non-prostate)',
    'Penis': 'Genitourinary (non-prostate)',
    'Renal Pelvis/Ureter': 'Genitourinary (non-prostate)',
    'Kidney': 'Genitourinary (non-prostate)',
    'Testis': 'Genitourinary (non-prostate)',
    'Urinary Bladder': 'Genitourinary (non-prostate)',
    'Adrenal (7th Ed)': 'Endocrine',
    'Thyroid - Anaplastic (7th Ed)': 'Endocrine',
    'Thyroid - Medullary (7th Ed)': 'Endocrine',
    'Thyroid - P/F under 45yrs (7th Ed)': 'Endocrine',
    'Thyroid - P/F 45 and older (7th Ed)': 'Endocrine',
    'Bone': 'Sarcoma',
    'Soft Tissue Sarcoma': 'Sarcoma',
    'Brain / Spinal Cord': 'Central Nervous System',
    'Unstageable Site': 'Unknown Primary', }

TUMOUR_SITE_ORDER = [
    'Breast',
    'Gastro-Intestinal',
    'Thoracic',
    'Prostate',
    'Gynaecologic',
    'Skin',
    'Haematologic',
    'Head and Neck',
    'Genitourinary (non-prostate)',
    'Endocrine',
    'Central Nervous System',
    'Sarcoma',
    'Unknown Primary',
    'Other',
]

import re

# Secondary lookup for cases where MajorGroupsStagingDesc == 'Unstageable Site'.
# Histology strings often embed ICD-O topography codes (e.g. C71._) that reveal
# the true anatomic site. Map of the C-code prefix to category:
HISTOLOGY_TOPOGRAPHY_MAP = {
    'C40': 'Sarcoma',                  # Bones, joints (long bones)
    'C41': 'Sarcoma',                  # Bones, joints (other / unspecified)
    'C47': 'Sarcoma',                  # Peripheral nerves / autonomic NS
    'C49': 'Sarcoma',                  # Connective / soft tissue
    'C70': 'Central Nervous System',   # Meninges
    'C71': 'Central Nervous System',   # Brain
    'C72': 'Central Nervous System',   # Spinal cord / other CNS
    'C53': 'Gynaecologic',             # Cervix uteri
    'C54': 'Gynaecologic',             # Corpus uteri
    'C56': 'Gynaecologic',             # Ovary
    'C57': 'Gynaecologic',             # Other female genital
}

# Fallback: histology-name keywords when no ICD-O code is embedded.
# Order matters — first match wins, so put more specific terms before generic ones.
HISTOLOGY_NAME_PATTERNS = [
    ('oligodendroglioma', 'Central Nervous System'),
    ('astrocytoma',       'Central Nervous System'),
    ('ependymoma',        'Central Nervous System'),
    ('medulloblastoma',   'Central Nervous System'),
    ('meningioma',        'Central Nervous System'),
    ('glioma',            'Central Nervous System'),
    ('glioblastoma',      'Central Nervous System'),
    ('osteosarcoma',      'Sarcoma'),
    ('rhabdomyosarcoma',  'Sarcoma'),
    ('leiomyosarcoma',    'Sarcoma'),
    ('liposarcoma',       'Sarcoma'),
    ('gastrointestinal stromal', 'Sarcoma'),
    ('gist',              'Sarcoma'),
    ('solitary fibrous tumor', 'Sarcoma'),
    ('fibrous histiocytoma',   'Sarcoma'),
    ('ewing',             'Sarcoma'),
    ('sarcoma',           'Sarcoma'),  # generic - keep last in the sarcoma group
    ('lymphoma',          'Haematologic'),
    ('leukemia',          'Haematologic'),
    ('leukaemia',         'Haematologic'),
    ('myeloma',           'Haematologic'),
    ('plasmacytoma',      'Haematologic'),
    ('melanoma',          'Skin'),
    ('mullerian mixed',   'Gynaecologic'),
]

ICDO_TOPOGRAPHY_PATTERN = re.compile(r'C(\d{2})\.')


def classify_unstageable_via_histology(histology_str):
    """Return a category for an 'Unstageable Site' case based on its histology
    string, or None if no confident reclassification is possible."""
    if not isinstance(histology_str, str) or not histology_str.strip():
        return None

    # Pass 1: embedded ICD-O topography code (most reliable)
    match = ICDO_TOPOGRAPHY_PATTERN.search(histology_str)
    if match:
        code = 'C' + match.group(1)
        if code in HISTOLOGY_TOPOGRAPHY_MAP:
            return HISTOLOGY_TOPOGRAPHY_MAP[code]

    # Pass 2: keyword match on histology name
    lowered = histology_str.lower()
    for pattern, category in HISTOLOGY_NAME_PATTERNS:
        if pattern in lowered:
            return category

    return None  # truly unclassifiable - leave as Unstageable / Unknown Primary

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

def summarize_tumour_site_outcomes(raw_df, histology_col=None):
    """Print N and number of deaths per tumour site category. Diagnostic only —
    does not write to disk."""
    tumour = raw_df['MajorGroupsStagingDesc'].fillna('').astype(str).str.strip()
    mapped = tumour.map(lambda value: TUMOUR_SITE_MAP.get(value, 'Other'))

    status = raw_df['PatStatus'].astype(str).str.strip()
    dead_mask = (status == 'D')

    summary = pd.DataFrame({
        'category': mapped,
        'dead': dead_mask.astype(int),
    })

    grouped = summary.groupby('category').agg(
        n=('dead', 'size'),
        deaths=('dead', 'sum'),
    )
    grouped['pct_of_cohort'] = (grouped['n'] * 100 / len(raw_df)).round(2)
    grouped['mortality_pct'] = (grouped['deaths'] * 100 / grouped['n']).round(1)

    # Reindex to your defined order, then append any categories that somehow
    # weren't in TUMOUR_SITE_ORDER (shouldn't happen, but a safety net)
    ordered = grouped.reindex(TUMOUR_SITE_ORDER)
    leftover = grouped.drop(index=TUMOUR_SITE_ORDER, errors='ignore')
    final = pd.concat([ordered, leftover])

    total_row = pd.DataFrame({
        'n': [len(raw_df)],
        'deaths': [int(dead_mask.sum())],
        'pct_of_cohort': [100.0],
        'mortality_pct': [round(dead_mask.sum() * 100 / len(raw_df), 1)],
    }, index=['TOTAL'])
    final = pd.concat([final, total_row])

    print("Tumour site N and deaths:")
    print(final.to_string())
    return final

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

def add_social_isolation_score(raw_df):
    """Add a composite social isolation count (0–5) and a count of observed
    domains for each patient."""
    social_df = raw_df[['LiveAlone', 'LostSpousePartner', 'Assistance',
                        'RegularContact', 'EmotionalSupport']].astype(str).apply(
        lambda s: s.str.strip()
    )

    missing_values = ['B', '99', '99.0', 'blank!', 'nan', 'NaN', '']

    domain_positive = pd.DataFrame({
        'LiveAlone':         social_df['LiveAlone'].isin(['Y', '1', '1.0']),
        'LostSpousePartner': social_df['LostSpousePartner'].isin(['Y', '1', '1.0']),
        'Assistance':        social_df['Assistance'].isin(['N', '0', '0.0']),
        'RegularContact':    social_df['RegularContact'].isin(['N', '0', '0.0']),
        'EmotionalSupport':  social_df['EmotionalSupport'].isin(['N', '0', '0.0']),
    })

    domain_observed = pd.DataFrame({
        col: ~social_df[col].isin(missing_values)
        for col in social_df.columns
    })

    raw_df['social_isolation_score']      = domain_positive.sum(axis=1).astype(int)
    raw_df['social_isolation_n_observed'] = domain_observed.sum(axis=1).astype(int)

    return raw_df

def build_patient_characteristics_table(pts_csv=RAW_PRISM_CSV, data_dir=DATA_DIR):
    output_path = f'{data_dir}patient_characteristics_table.csv'
    raw_df = pd.read_csv(pts_csv, index_col=INDEX, engine='python', encoding='latin1')

    unmapped = raw_df.loc[
    ~raw_df['MajorGroupsStagingDesc'].isin(TUMOUR_SITE_MAP.keys()),'MajorGroupsStagingDesc'].value_counts(dropna=False)
    print("UNMAPPED TUMOUR SITES:")
    print(unmapped)

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

    raw_df = add_social_isolation_score(raw_df)

    female_n = int((raw_df['Sex'] == 'F').sum())
    male_n = int((raw_df['Sex'] == 'M').sum())
    alive_n = int((raw_df['PatStatus'] == 'A').sum())

    append_row(rows, 'Patient characteristics', 'Female', female_n, percent_non_missing(female_n, n_pts, 0), 0)
    append_row(rows, 'Patient characteristics', 'Male', male_n, percent_non_missing(male_n, n_pts, 0), 0)
    append_row(rows, 'Patient characteristics', 'Alive at last follow-up', alive_n, percent_non_missing(alive_n, n_pts, 0), 0)

# Performance Status (ECOG)
    # Blank values, 88, and 99 are all coded as unknown.
    ps_numeric = pd.to_numeric(raw_df['PerformanceStatus'], errors='coerce')

    ps_missing_mask = ps_numeric.isna() | ps_numeric.isin([88, 99])
    ps_missing_n = int(ps_missing_mask.sum())

    ps_categories = [
        ('0 - Asymptomatic',                          0),
        ('1 - Symptomatic, fully ambulatory',         1),
        ('2 - Symptomatic, in bed <50% of day',       2),
        ('3 - Symptomatic, in bed >50% of day',       3),
        ('4 - Bedbound',                              4),
        ('5 - Dead',                                  5),
    ]

    ps_category_total = 0
    for label, val in ps_categories:
        count = int((ps_numeric == val).sum())
        ps_category_total += count
        append_row(
            rows,
            'Performance status (ECOG)',
            label,
            count,
            percent_non_missing(count, n_pts, ps_missing_n),
            ps_missing_n,
        )

    append_row(
        rows,
        'Performance status (ECOG)',
        'Unknown',
        ps_missing_n,
        round(ps_missing_n * 100 / n_pts, 2),
        ps_missing_n,
    )

    # Sanity check: categories + unknown should equal total cohort
    if ps_category_total + ps_missing_n != n_pts:
        print(f"WARNING: PerformanceStatus rows sum to {ps_category_total + ps_missing_n}, "
              f"expected {n_pts}. Unrecognised values:")
        unrecognised = ps_numeric[
            ~ps_numeric.isin([0, 1, 2, 3, 4, 5]) & ~ps_missing_mask
        ]
        print(unrecognised.value_counts(dropna=False))
    
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
        missing_n = int(raw_col.isin(['B', '99', '99.0', 'blank!', 'nan', '88', '88.0', '88.00']).sum())
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

    # Composite social isolation score (count of positive domains, 0–5)
    score = raw_df['social_isolation_score']
    n_observed = raw_df['social_isolation_n_observed']

    eligible_mask = n_observed > 0
    n_eligible = int(eligible_mask.sum())
    n_no_data = n_pts - n_eligible

    append_row(
        rows,
        'Composite social isolation score',
        'Mean (SD)',
        mean=round(score[eligible_mask].mean(), 2),
        sd=round(score[eligible_mask].std(), 2),
    )
    append_row(
        rows,
        'Composite social isolation score',
        'Median',
        median=int(score[eligible_mask].median()),
    )

    for k in range(6):
        count_k = int(((score == k) & eligible_mask).sum())
        append_row(
            rows,
            'Composite social isolation score',
            f'{k} positive domain{"s" if k != 1 else ""}',
            count_k,
            percent_non_missing(count_k, n_pts, n_no_data),
            n_no_data,
        )

    append_row(
        rows,
        'Composite social isolation score',
        'No domains observed (excluded)',
        n_no_data,
        round(n_no_data * 100 / n_pts, 2),
        n_no_data,
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
    filename = f'{data_dir}dataset_analysis_supplementary.csv'

    # Check if base preprocessed pts file already generated, return if so unless overwrite=True
    if path.exists(filename) and not overwrite:
        print("dataset_analysis_supplementary.csv already generated, will load and exit preprocesser")
        return pd.read_csv(filename, index_col=INDEX)

    # Otherwise, generate
    print("dataset_analysis_supplementary.csv is being generated, this may take a few minutes")
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
    raw_df = pd.read_csv(RAW_PRISM_CSV, index_col=INDEX, engine='python', encoding='latin1')
    summarize_tumour_site_outcomes(raw_df)
    analyze_dataset(overwrite=True)
    build_patient_characteristics_table()
    print("Analysis complete!")


