"""
Combined multivariable Cox proportional hazards analysis.

For JCO Oncology Practice resubmission.

A SINGLE multivariable model is fit with all predictors entered
simultaneously (mutual adjustment), rather than one model per predictor:

  Predictors of interest (entered together):
    - 5 social isolation domains: lives alone, recent partner loss,
      no regular social contact, no emotional support, no IADL assistance
    - 2 psychological distress variables: depression, anxiety
  Confounders:
    - age, sex, metastatic status

Two specifications of tumour group:
  PRIMARY (stratified):    tumour group as strata -> separate baseline
                           hazard per group, NO HR reported for cancer type.
  SENSITIVITY (covariate): tumour group as dummy covariates
                           (reference = Breast) -> HR per cancer type
                           appears in the table.
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
    'SK':  'Skin (Non-melanoma)',
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

# Reference level for the tumour-group covariate specification.
TUMOUR_GROUP_REFERENCE = 'Breast'
# Catch-all bucket for any tumour group too small to estimate on its own.
TUMOUR_GROUP_OTHER = 'Other / Unknown'
# Minimum patients per tumour group before folding into the catch-all.
MIN_GROUP_N = 20


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

    # Sex: match on the leading letter so 'F', 'f', 'Female' all resolve to
    # female; anything else is treated as not-female. Robust to coding style.
    sex_str = df['Sex'].astype(str).str.strip().str.upper()
    df['female'] = sex_str.str.startswith('F').astype(int)

    tg = df['TumourGroup'].fillna('').astype(str).str.strip().str.upper()
    df['tumour_group'] = tg.map(lambda v: TUMOUR_GROUP_MAP.get(v, TUMOUR_GROUP_OTHER))

    # Metastatic status. M1_IV_ALL_CAT is coded 0 / 1 / 99, with 1 = metastatic.
    # Coerce to NUMERIC before mapping: if the column is read as float
    # (e.g. 1.0 / 0.0 / 99.0 because of missing values), a string comparison
    # silently sends every row to the non-metastatic branch. Compare numerically.
    # 1 -> Metastatic; 0, 99, and missing all map to Non-metastatic. This is a
    # strictly binary variable -- there is no separate 'Unknown' level.
    met_num = pd.to_numeric(df['M1_IV_ALL_CAT'], errors='coerce')
    df['metastatic'] = np.where(met_num == 1, 'Metastatic', 'Non-metastatic')

    # Cohort restriction
    df = df.dropna(subset=['observed_mo', 'died', 'age']).copy()
    df = df[df['observed_mo'] >= 0].copy()

    # Fold any tumour group below MIN_GROUP_N into the catch-all bucket so a
    # sparse group is never silently absorbed into the covariate reference.
    counts = df['tumour_group'].value_counts()
    rare = counts[counts < MIN_GROUP_N].index.tolist()
    if rare:
        print(f"  Folding sparse tumour groups (n<{MIN_GROUP_N}) into "
              f"'{TUMOUR_GROUP_OTHER}': {rare}")
        df.loc[df['tumour_group'].isin(rare), 'tumour_group'] = TUMOUR_GROUP_OTHER

    _sanity_check_coding(df)

    return df


def _sanity_check_coding(df):
    """Loud diagnostics so a silent miscoding cannot pass unnoticed."""
    print("\n  --- coding sanity check ---")
    print("  Sex (raw) -> female mean:", round(float(df['female'].mean()), 3))
    print("  Raw M1_IV_ALL_CAT value counts:")
    print(df['M1_IV_ALL_CAT'].value_counts(dropna=False).to_string()
          .replace('\n', '\n    ').rjust(0))
    met_levels = df['metastatic'].value_counts()
    print("  Derived metastatic levels:", met_levels.to_dict())
    if met_levels.shape[0] < 2:
        print("  *** WARNING: metastatic has a single level -> it will NOT "
              "enter the model. Check M1_IV_ALL_CAT coding. ***")
    if df['female'].nunique() < 2:
        print("  *** WARNING: 'female' is constant -> sex will NOT enter the "
              "model. Check the Sex coding. ***")
    print("  ---------------------------\n")


def build_design_matrix(df, predictor_cols, mode):
    """Build the design matrix for the combined model.

    All predictors in `predictor_cols` are entered together. `mode`
    controls only how tumour group is handled:
      'stratified' -> tumour_group kept as a column for use as strata
      'covariate'  -> tumour_group expanded to dummies (ref = Breast)
    """
    predictor_cols = list(predictor_cols)

    keep = (['observed_mo', 'died'] + predictor_cols
            + ['age', 'female', 'metastatic', 'tumour_group'])
    sub = df[keep].copy()

    # Coerce predictors to numeric and require strict 0/1. A non-binary
    # column would make exp(coef) a per-unit HR rather than the intended
    # exposed-vs-unexposed contrast, so fail loudly instead.
    for pc in predictor_cols:
        vals = pd.to_numeric(sub[pc], errors='coerce')
        observed = set(np.unique(vals.dropna().to_numpy()))
        if not observed <= {0, 1}:
            raise ValueError(
                f"Predictor {pc!r} is not binary 0/1; observed values "
                f"{sorted(observed)[:8]}. HRs would be misinterpreted."
            )
        sub[pc] = vals

    # One-hot metastatic status, drop the reference level.
    sub = pd.get_dummies(
        sub, columns=['metastatic'], drop_first=False, dtype=int
    )
    if 'metastatic_Non-metastatic' in sub.columns:
        sub = sub.drop(columns=['metastatic_Non-metastatic'])

    # Tumour group encoding for the sensitivity (covariate) model.
    if mode == 'covariate':
        sub = pd.get_dummies(
            sub, columns=['tumour_group'], drop_first=False, dtype=int
        )
        ref_col = f'tumour_group_{TUMOUR_GROUP_REFERENCE}'
        if ref_col in sub.columns:
            sub = sub.drop(columns=[ref_col])

    # Ensure numeric columns are numeric; tumour_group (strata) stays as-is.
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


def extract_hr(cph_summary, term):
    if term not in cph_summary.index:
        return {'HR': np.nan, 'lo': np.nan, 'hi': np.nan, 'p': np.nan}

    row = cph_summary.loc[term]
    return {
        'HR': row['exp(coef)'],
        'lo': row['exp(coef) lower 95%'],
        'hi': row['exp(coef) upper 95%'],
        'p':  row['p'],
    }


def fit_combined_model(df, predictor_cols, mode, results_dir, label):
    """Fit one combined multivariable Cox model and persist its summary."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print('='*70)

    try:
        design = build_design_matrix(df, predictor_cols, mode=mode)
        cph = fit_cox(design, mode=mode)
    except (ConvergenceError, np.linalg.LinAlgError, ValueError) as e:
        print(f"  FIT FAILED ({type(e).__name__})")
        print(f"    {str(e)[:300]}")
        return None, 0, 0

    n = len(design)
    events = int(design['died'].sum())
    print(f"  N = {n:,}   events = {events:,}")

    # Confirm the confounders actually made it into the model.
    for needed in ('metastatic_Metastatic', 'female', 'age'):
        present = needed in cph.summary.index
        print(f"    term in model: {needed:24s} {'yes' if present else 'NO'}")

    cols = ['coef', 'exp(coef)',
            'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']
    print(cph.summary[cols].round(4).to_string())

    cph.summary.to_csv(os.path.join(results_dir, f'cox_combined_{mode}.csv'))

    return cph, n, events


def build_predictor_table(cph_strat, cph_cov, predictor_cols):
    """Side-by-side HRs for the predictors of interest from both models."""
    rows = []
    for p in predictor_cols:
        primary = extract_hr(cph_strat.summary, p) if cph_strat is not None \
            else {'HR': np.nan, 'lo': np.nan, 'hi': np.nan, 'p': np.nan}
        sens = extract_hr(cph_cov.summary, p) if cph_cov is not None \
            else {'HR': np.nan, 'lo': np.nan, 'hi': np.nan, 'p': np.nan}

        rows.append({
            'predictor':          p,
            'display':            PREDICTOR_DISPLAY.get(p, p),
            'HR_primary':         primary['HR'],
            'HR_primary_lo':      primary['lo'],
            'HR_primary_hi':      primary['hi'],
            'p_primary':          primary['p'],
            'HR_sensitivity':     sens['HR'],
            'HR_sensitivity_lo':  sens['lo'],
            'HR_sensitivity_hi':  sens['hi'],
            'p_sensitivity':      sens['p'],
        })

    return pd.DataFrame(rows)


def build_tumour_group_table(cph_cov):
    """HR per cancer type vs Breast, from the covariate model only."""
    if cph_cov is None:
        return pd.DataFrame()

    rows = []
    for term in cph_cov.summary.index:
        if not term.startswith('tumour_group_'):
            continue
        hr = extract_hr(cph_cov.summary, term)
        rows.append({
            'tumour_group': term.replace('tumour_group_', ''),
            'reference':    TUMOUR_GROUP_REFERENCE,
            'HR':           hr['HR'],
            'lo':           hr['lo'],
            'hi':           hr['hi'],
            'p':            hr['p'],
        })

    return pd.DataFrame(rows)


def _round_hr(table, hr_cols):
    for col in hr_cols:
        if col in table.columns:
            table[col] = table[col].round(3)
    return table


def _format_p(table, p_cols):
    for col in p_cols:
        if col in table.columns:
            table[col] = table[col].apply(
                lambda x: f"{x:.4g}" if pd.notna(x) else "FAILED"
            )
    return table


def main(results_dir=RESULTS_DIR):
    out_dir = os.path.join(results_dir, 'cox_combined')
    os.makedirs(out_dir, exist_ok=True)

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

    predictor_cols = list(SOC_ISO_FEATS) + list(PSYCH_DIST_FEATS)

    cph_strat, n_strat, ev_strat = fit_combined_model(
        df, predictor_cols, 'stratified', out_dir,
        'PRIMARY: combined model, stratified by tumour group'
    )

    cph_cov, n_cov, ev_cov = fit_combined_model(
        df, predictor_cols, 'covariate', out_dir,
        'SENSITIVITY: combined model, tumour group as covariate '
        f'(reference = {TUMOUR_GROUP_REFERENCE})'
    )

    # Predictors of interest, both specifications side by side.
    predictor_table = build_predictor_table(cph_strat, cph_cov, predictor_cols)
    predictor_table = _round_hr(
        predictor_table,
        ['HR_primary', 'HR_primary_lo', 'HR_primary_hi',
         'HR_sensitivity', 'HR_sensitivity_lo', 'HR_sensitivity_hi']
    )
    predictor_table = _format_p(predictor_table, ['p_primary', 'p_sensitivity'])
    predictor_table.to_csv(
        os.path.join(out_dir, 'cox_combined_predictors.csv'), index=False
    )

    # Tumour-group HRs (covariate model only).
    tumour_table = build_tumour_group_table(cph_cov)
    tumour_table = _round_hr(tumour_table, ['HR', 'lo', 'hi'])
    tumour_table = _format_p(tumour_table, ['p'])
    tumour_table.to_csv(
        os.path.join(out_dir, 'cox_combined_tumour_group.csv'), index=False
    )

    print(f"\n{'='*70}")
    print("PREDICTORS OF INTEREST  (mutually adjusted)")
    print('='*70)
    print(predictor_table.to_string(index=False))

    print(f"\n{'='*70}")
    print(f"TUMOUR GROUP HRs vs {TUMOUR_GROUP_REFERENCE}  (sensitivity / covariate model)")
    print('='*70)
    if tumour_table.empty:
        print("  (covariate model did not fit; no tumour-group HRs available)")
    else:
        print(tumour_table.to_string(index=False))

    print(f"\nAll outputs written to: {out_dir}/")
    return predictor_table, tumour_table


if __name__ == '__main__':
    main()


