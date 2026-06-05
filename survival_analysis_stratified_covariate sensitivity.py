"""
Cox proportional hazards analyses for individual social isolation domains
and psychological distress variables.

For JCO Oncology Practice resubmission.

For each of 7 binary predictors (5 social isolation domains + 2 distress vars):
  PRIMARY:     Cox PH stratified by tumour group, adjusted for
               age + sex + metastatic status.
  SENSITIVITY: Same model but tumour group as categorical covariate
               (reference = Breast) instead of strata.
"""

import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError

from preprocesser import preprocess
from globals import (
    SOC_ISO_FEATS, PSYCH_DIST_FEATS, RESULTS_DIR,
    RAW_PRISM_CSV, INDEX,
)


TUMOUR_GROUP_MAP = {
    'BR':  'Breast',
    'LU':  'Lung',
    'GI':  'Gastrointestinal',
    'GIG': 'Gastrointestinal',
    'SAG': 'Gastrointestinal',
    'GU':  'Genitourinary',
    'GY':  'Gynecologic',
    'HN':  'Head and Neck',
    'LY':  'Lymphoma',
    'ME':  'Melanoma',
    'SK':  'Skin',
    'SA':  'Sarcoma',
    'NE':  'Central Nervous System',
    'LK':  'Leukemia',
    'OO':  'Other / Unknown',
    'OT':  'Other / Unknown',
    'PU':  'Other / Unknown',
    'UN':  'Other / Unknown',
    'UNK': 'Other / Unknown',
}

PREDICTOR_DISPLAY = {
    'LiveAlone':            'Lives alone',
    'LostSpousePartner':    'Recent partner loss',
    'NoRegularContact':     'No regular social contact',
    'NoEmotionalSupport':   'No emotional support',
    'NoAssistance':         'No IADL assistance',
    'DepressionScore_CAT2': 'Depression symptoms',
    'AnxietyScore_CAT2':    'Anxiety symptoms',
}


def prepare_analysis_df(df):
    """Merge confounders from the raw CSV into the preprocessed df."""
    df = df.copy()

    raw_confounders = pd.read_csv(
        RAW_PRISM_CSV, index_col=INDEX, engine='python', encoding='latin1',
        usecols=lambda c: c in (
            INDEX, 'AgeAtDiagnosis', 'Sex',
            'M1_IV_ALL_CAT', 'TumourGroup'
        )
    )
    df = df.join(raw_confounders, how='left', rsuffix='_raw')

    df['age'] = pd.to_numeric(df['AgeAtDiagnosis'], errors='coerce')
    df['female'] = (df['Sex'].astype(str).str.strip() == 'F').astype(int)

    tg = df['TumourGroup'].fillna('').astype(str).str.strip().str.upper()
    df['tumour_group'] = tg.map(lambda v: TUMOUR_GROUP_MAP.get(v, 'Other / Unknown'))

    # Metastatic: fold unknown into non-metastatic
    met = df['M1_IV_ALL_CAT'].astype(str).str.strip()
    df['metastatic'] = met.map(
        {'1': 'Metastatic', '0': 'Non-metastatic', '99': 'Non-metastatic'}
    ).fillna('Non-metastatic')

    # Cohort restriction
    df = df.dropna(subset=['observed_mo', 'died', 'age']).copy()
    df = df[df['observed_mo'] >= 0].copy()

    return df


def build_design_matrix(df, predictor_col, mode, min_count=20):
    """Build the design matrix for one predictor."""

    keep = ['observed_mo', 'died', predictor_col,
            'age', 'female', 'metastatic', 'tumour_group']

    sub = df[keep].copy()

    # Ensure binary predictor
    sub[predictor_col] = pd.to_numeric(sub[predictor_col], errors='coerce').astype(int)

    # One-hot metastatic status
    sub = pd.get_dummies(
        sub, columns=['metastatic'],
        drop_first=False, dtype=int
    )

    # Drop reference level
    if 'metastatic_Non-metastatic' in sub.columns:
        sub = sub.drop(columns=['metastatic_Non-metastatic'])

    # Tumour group encoding for sensitivity model
    if mode == 'covariate':
        sub = pd.get_dummies(
            sub, columns=['tumour_group'],
            drop_first=False, dtype=int
        )
        if 'tumour_group_Breast' in sub.columns:
            sub = sub.drop(columns=['tumour_group_Breast'])

    # Drop sparse dummies
    protected = {'observed_mo', 'died', 'age', 'female',
                 predictor_col, 'tumour_group'}

    dropped = []
    for c in list(sub.columns):
        if c in protected or c.startswith('metastatic_'):
            continue
        if sub[c].dtype == object:
            continue
        if sub[c].sum() < min_count:
            dropped.append((c, int(sub[c].sum())))
            sub = sub.drop(columns=[c])

    if dropped:
        print(f"    Dropped sparse dummies (n<{min_count}):")
        for col, n in dropped:
            print(f"      {col}: n={n}")

    # Ensure numeric
    numeric_cols = [c for c in sub.columns if c != 'tumour_group']
    for c in numeric_cols:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')

    sub = sub.dropna(subset=numeric_cols)

    return sub


def fit_cox(design_df, mode):
    cph = CoxPHFitter()

    if mode == 'stratified':
        cph.fit(
            design_df,
            duration_col='observed_mo',
            event_col='died',
            strata=['tumour_group'],
            robust=True,
            show_progress=False
        )
    else:
        cph.fit(
            design_df.drop(columns=['tumour_group'], errors='ignore'),
            duration_col='observed_mo',
            event_col='died',
            robust=True,
            show_progress=False
        )

    return cph


def extract_predictor_hr(cph_summary, predictor_col):
    if predictor_col not in cph_summary.index:
        return {'HR': np.nan, 'lo': np.nan, 'hi': np.nan, 'p': np.nan}

    row = cph_summary.loc[predictor_col]

    return {
        'HR': row['exp(coef)'],
        'lo': row['exp(coef) lower 95%'],
        'hi': row['exp(coef) upper 95%'],
        'p':  row['p'],
    }


def safe_fit_and_extract(df, predictor_col, mode, results_dir, label):
    nan_hr = {'HR': np.nan, 'lo': np.nan, 'hi': np.nan, 'p': np.nan}

    try:
        design = build_design_matrix(df, predictor_col, mode=mode)
        cph = fit_cox(design, mode=mode)

        print(f"\n  {label}:")
        print(cph.summary[['coef', 'exp(coef)',
                           'exp(coef) lower 95%', 'exp(coef) upper 95%',
                           'p']].round(4))

        cph.summary.to_csv(
            os.path.join(results_dir, f'cox_{predictor_col}_{mode}.csv')
        )

        n = len(design)
        events = int(design['died'].sum())

        return cph, extract_predictor_hr(cph.summary, predictor_col), n, events

    except (ConvergenceError, np.linalg.LinAlgError, ValueError) as e:
        print(f"\n  {label}: FIT FAILED ({type(e).__name__})")
        print(f"    {str(e)[:300]}")
        return None, nan_hr, 0, 0


def run_cox_for_predictor(df, predictor_col, results_dir):
    display = PREDICTOR_DISPLAY.get(predictor_col, predictor_col)

    print(f"\n{'='*70}")
    print(f"  {display}  ({predictor_col})")
    print('='*70)

    _, hr_primary, n_strat, ev_strat = safe_fit_and_extract(
        df, predictor_col, 'stratified', results_dir,
        'Primary (stratified by tumour group)'
    )

    _, hr_sensitivity, n_cov, ev_cov = safe_fit_and_extract(
        df, predictor_col, 'covariate', results_dir,
        'Sensitivity (tumour group as covariate, reference = Breast)'
    )

    return {
        'predictor': predictor_col,
        'display': display,
        'n_strat': n_strat,
        'events_strat': ev_strat,
        'n_cov': n_cov,
        'events_cov': ev_cov,
        'HR_primary': hr_primary['HR'],
        'HR_primary_lo': hr_primary['lo'],
        'HR_primary_hi': hr_primary['hi'],
        'p_primary': hr_primary['p'],
        'HR_sensitivity': hr_sensitivity['HR'],
        'HR_sensitivity_lo': hr_sensitivity['lo'],
        'HR_sensitivity_hi': hr_sensitivity['hi'],
        'p_sensitivity': hr_sensitivity['p'],
    }


def main(results_dir=RESULTS_DIR):
    cox_dir = os.path.join(results_dir, 'cox_domains')
    os.makedirs(cox_dir, exist_ok=True)

    print("Loading and preprocessing data...")
    df = preprocess()
    df = prepare_analysis_df(df)

    print(f"\n=== Analytic cohort ===")
    print(f"N:                  {len(df):,}")
    print(f"Total deaths:       {int(df['died'].sum()):,}")
    print(f"Median follow-up:   {df['observed_mo'].median():.1f} months")

    print(f"\nMetastatic status distribution:")
    print(df['metastatic'].value_counts().to_string())

    print(f"\nTumour group counts:")
    print(df['tumour_group'].value_counts().to_string())

    summary_rows = []
    all_predictors = SOC_ISO_FEATS + PSYCH_DIST_FEATS

    for predictor in all_predictors:
        result = run_cox_for_predictor(df, predictor, cox_dir)
        summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows)

    for col in ['HR_primary', 'HR_primary_lo', 'HR_primary_hi',
                'HR_sensitivity', 'HR_sensitivity_lo', 'HR_sensitivity_hi']:
        summary_df[col] = summary_df[col].round(3)

    for col in ['p_primary', 'p_sensitivity']:
        summary_df[col] = summary_df[col].apply(
            lambda x: f"{x:.4g}" if pd.notna(x) else "FAILED"
        )

    summary_path = os.path.join(cox_dir, 'cox_domains_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)

    print(summary_df.to_string(index=False))

    print(f"\nAll outputs written to: {cox_dir}/")
    return summary_df


if __name__ == '__main__':
    main()


