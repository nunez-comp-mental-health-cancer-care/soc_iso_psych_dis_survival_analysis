"""
Survival analysis for the composite social isolation score.

Addresses the reviewer comment on dose-response.

(1) Kaplan-Meier survival curves by social isolation score (0, 1, 2, 3+).
(2) Cox PH models:
      PRIMARY:     stratified by tumour group; score as categorical (ref = 0)
                   AND as a linear ordinal term (per +1).
      SENSITIVITY: same models with tumour group as a categorical covariate
                   (reference = Breast) instead of strata.
(3) Likelihood-ratio tests for dose-response:
      - linear trend (ordinal score vs no score)
      - departure from linearity (categorical vs linear)
      - overall score effect (categorical vs no score)

Robust (sandwich) standard errors are used for the reported HRs/CIs. The
dose-response LRTs are computed from the partial log-likelihoods of the
stratified (primary) models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.exceptions import ConvergenceError
from lifelines.utils import median_survival_times
from lifelines.plotting import add_at_risk_counts

from globals import RAW_PRISM_CSV, RESULTS_DIR, INDEX


# Labels aligned to the manuscript (Skin (Non-melanoma); Other / Unknown).
TUMOUR_GROUP_MAP = {
    'BR': 'Breast', 'LU': 'Lung',
    'GI': 'Gastrointestinal', 'GIG': 'Gastrointestinal', 'SAG': 'Gastrointestinal',
    'GU': 'Genitourinary', 'GY': 'Gynecologic', 'HN': 'Head and Neck',
    'LY': 'Lymphoma', 'ME': 'Melanoma', 'SK': 'Skin (Non-melanoma)', 'SA': 'Sarcoma',
    'NE': 'Central Nervous System', 'LK': 'Leukemia',
    'OO': 'Other / Unknown', 'OT': 'Other / Unknown',
    'PU': 'Other / Unknown', 'UN': 'Other / Unknown', 'UNK': 'Other / Unknown',
}

TUMOUR_GROUP_REFERENCE = 'Breast'
TUMOUR_GROUP_OTHER = 'Other / Unknown'
MIN_GROUP_N = 20

SCORE_CAT_ORDER = ['0', '1', '2', '3plus']
SCORE_CAT_LABEL = {'0': 'Score 0', '1': 'Score 1', '2': 'Score 2', '3plus': 'Score 3+'}


# =============================
# DATA PREP
# =============================

def add_social_isolation_score(raw_df):
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

    raw_df['social_isolation_score'] = domain_positive.sum(axis=1).astype(int)
    raw_df['social_isolation_n_observed'] = domain_observed.sum(axis=1).astype(int)
    return raw_df


def prepare_survival_dataframe(raw_df):
    df = raw_df.copy()
    df = add_social_isolation_score(df)

    df['time_months'] = pd.to_numeric(df['Dx_to_death_lastfu_months'], errors='coerce')
    df['event'] = (df['PatStatus'].astype(str).str.strip() == 'D').astype(int)

    df['score_cat'] = df['social_isolation_score'].apply(
        lambda x: '3plus' if x >= 3 else str(int(x))
    )
    # Linear ordinal term for the trend test. The top category is collapsed at
    # 3 to match the categorical coding (0, 1, 2, 3+), so the linear model is
    # strictly nested in the categorical model and the departure-from-linearity
    # LRT is valid. (Previously this used the uncapped 0-5 score; capping at 3
    # aligns the trend term with the 3+ category. One-line revert if undesired.)
    df['score_linear'] = df['social_isolation_score'].clip(upper=3).astype(int)

    df['age'] = pd.to_numeric(df['AgeAtDiagnosis'], errors='coerce')

    # Sex by leading letter so 'F'/'f'/'Female' resolve to female; robust to
    # coding style (matches the primary model).
    sex_str = df['Sex'].astype(str).str.strip().str.upper()
    df['female'] = sex_str.str.startswith('F').astype(int)

    tg = df['TumourGroup'].fillna('').astype(str).str.strip().str.upper()
    df['tumour_group'] = tg.map(lambda v: TUMOUR_GROUP_MAP.get(v, TUMOUR_GROUP_OTHER))

    # Metastatic status. M1_IV_ALL_CAT is coded 0 / 1 / 99, with 1 = metastatic.
    # Coerce to NUMERIC before mapping: if the column is read as float
    # (1.0 / 0.0 / 99.0 because of missing values), a string comparison
    # silently sends every row to the non-metastatic branch. Compare numerically.
    # 1 -> Metastatic; 0, 99, and missing all map to Non-metastatic. Strictly
    # binary -- no separate 'Unknown' level.
    met_num = pd.to_numeric(df['M1_IV_ALL_CAT'], errors='coerce')
    df['metastatic'] = np.where(met_num == 1, 'Metastatic', 'Non-metastatic')

    df = df[df['social_isolation_n_observed'] > 0].copy()
    df = df.dropna(subset=['time_months', 'event', 'age']).copy()
    df = df[df['time_months'] >= 0].copy()

    # Fold any tumour group below MIN_GROUP_N into the catch-all so a sparse
    # group is never silently absorbed into the covariate reference.
    counts = df['tumour_group'].value_counts()
    rare = counts[counts < MIN_GROUP_N].index.tolist()
    if rare:
        print(f"  Folding sparse tumour groups (n<{MIN_GROUP_N}) into "
              f"'{TUMOUR_GROUP_OTHER}': {rare}")
        df.loc[df['tumour_group'].isin(rare), 'tumour_group'] = TUMOUR_GROUP_OTHER

    print("\n  --- coding sanity check ---")
    print("  female mean:", round(float(df['female'].mean()), 3))
    print("  metastatic levels:", df['metastatic'].value_counts().to_dict())
    if df['metastatic'].nunique() < 2:
        print("  *** WARNING: metastatic has a single level -> will NOT enter "
              "the model. Check M1_IV_ALL_CAT coding. ***")
    print("  score_cat distribution:", df['score_cat'].value_counts().to_dict())
    print("  ---------------------------")

    return df


# =============================
# DESIGN MATRIX
# =============================

def build_design_matrix(model_df, score_col, mode, ref_levels):
    """score_col may be 'score_cat', 'score_linear', or None (no score term)."""
    keep = ['time_months', 'event']
    if score_col is not None:
        keep.append(score_col)
    keep += ['age', 'female', 'metastatic', 'tumour_group']

    sub = model_df[keep].copy()

    sub = pd.get_dummies(sub, columns=['metastatic'], drop_first=False, dtype=int)

    if score_col == 'score_cat':
        sub = pd.get_dummies(sub, columns=['score_cat'], drop_first=False, dtype=int)

    for ref_col in ref_levels:
        if ref_col in sub.columns:
            sub = sub.drop(columns=[ref_col])

    if mode == 'covariate':
        sub = pd.get_dummies(sub, columns=['tumour_group'], drop_first=False, dtype=int)
        ref_col = f'tumour_group_{TUMOUR_GROUP_REFERENCE}'
        if ref_col in sub.columns:
            sub = sub.drop(columns=[ref_col])

    numeric_cols = [c for c in sub.columns if c != 'tumour_group']
    for c in numeric_cols:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')

    sub = sub.dropna(subset=numeric_cols)
    return sub


# =============================
# MODEL FITTING
# =============================

def fit_cox(design_df, mode):
    cph = CoxPHFitter()
    if mode == 'stratified':
        cph.fit(design_df, duration_col='time_months', event_col='event',
                strata=['tumour_group'], robust=True)
    else:
        cph.fit(design_df.drop(columns=['tumour_group'], errors='ignore'),
                duration_col='time_months', event_col='event', robust=True)
    return cph


# =============================
# KAPLAN-MEIER
# =============================

def kaplan_meier_curves(df, output_dir, x_max=84):
    """KM by composite score. Conventions: x-axis truncated ~84 months,
    no log-rank p-value shown, median OS (95% CI) annotated, at-risk counts
    for every group."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    kmfs, med_rows = [], []

    for lev in SCORE_CAT_ORDER:
        mask = df['score_cat'] == lev
        if mask.sum() == 0:
            continue
        kmf = KaplanMeierFitter(label=SCORE_CAT_LABEL[lev])
        kmf.fit(df.loc[mask, 'time_months'], df.loc[mask, 'event'])
        kmf.plot_survival_function(ax=ax, ci_show=True)
        kmfs.append(kmf)

        med = kmf.median_survival_time_
        ci = median_survival_times(kmf.confidence_interval_)
        lo, hi = float(ci.iloc[0, 0]), float(ci.iloc[0, 1])
        med_rows.append({
            'group': SCORE_CAT_LABEL[lev], 'n': int(mask.sum()),
            'events': int(df.loc[mask, 'event'].sum()),
            'median_OS_months': med, 'median_lo': lo, 'median_hi': hi,
        })

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Months since diagnosis')
    ax.set_ylabel('Overall survival probability')
    ax.set_title('Overall survival by composite social isolation score')

    def _fmt(v):
        return 'NR' if (v is None or not np.isfinite(v)) else f'{v:.1f}'
    txt = 'Median OS (95% CI)\n' + '\n'.join(
        f"{r['group']}: {_fmt(r['median_OS_months'])} mo "
        f"({_fmt(r['median_lo'])}-{_fmt(r['median_hi'])})" for r in med_rows
    )
    ax.text(0.02, 0.04, txt, transform=ax.transAxes, fontsize=8,
            va='bottom', ha='left',
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))

    add_at_risk_counts(*kmfs, ax=ax)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'km_composite_score.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    med_df = pd.DataFrame(med_rows)
    med_df.to_csv(os.path.join(output_dir, 'km_medians.csv'), index=False)
    return med_df, fig_path


# =============================
# COX PIPELINE
# =============================

def cox_proportional_hazards(df, output_dir):
    REF_MET = 'metastatic_Non-metastatic'
    REF_SCORE_CAT = 'score_cat_0'
    common_refs = [REF_MET]

    def safe_fit(score_col, mode, ref_levels):
        try:
            design = build_design_matrix(df, score_col, mode, ref_levels)
            return fit_cox(design, mode)
        except (ConvergenceError, np.linalg.LinAlgError, ValueError) as e:
            print(f"  fit failed [{score_col}/{mode}]: {type(e).__name__}")
            return None

    cph_cat_strat = safe_fit('score_cat', 'stratified', common_refs + [REF_SCORE_CAT])
    cph_lin_strat = safe_fit('score_linear', 'stratified', common_refs)
    cph_cat_cov = safe_fit('score_cat', 'covariate', common_refs + [REF_SCORE_CAT])
    cph_lin_cov = safe_fit('score_linear', 'covariate', common_refs)

    def extract(cph, var):
        if cph is None or var not in cph.summary.index:
            return {'HR': np.nan, 'lo': np.nan, 'hi': np.nan, 'p': np.nan}
        row = cph.summary.loc[var]
        return {'HR': row['exp(coef)'], 'lo': row['exp(coef) lower 95%'],
                'hi': row['exp(coef) upper 95%'], 'p': row['p']}

    rows = []
    for var, label in [('score_cat_1', 'Score 1 vs 0'),
                       ('score_cat_2', 'Score 2 vs 0'),
                       ('score_cat_3plus', 'Score 3+ vs 0')]:
        prim, sens = extract(cph_cat_strat, var), extract(cph_cat_cov, var)
        rows.append({'comparison': label,
                     'HR_primary': prim['HR'], 'HR_primary_lo': prim['lo'],
                     'HR_primary_hi': prim['hi'], 'p_primary': prim['p'],
                     'HR_sensitivity': sens['HR'], 'HR_sensitivity_lo': sens['lo'],
                     'HR_sensitivity_hi': sens['hi'], 'p_sensitivity': sens['p']})

    prim, sens = extract(cph_lin_strat, 'score_linear'), extract(cph_lin_cov, 'score_linear')
    rows.append({'comparison': 'Per +1 increase',
                 'HR_primary': prim['HR'], 'HR_primary_lo': prim['lo'],
                 'HR_primary_hi': prim['hi'], 'p_primary': prim['p'],
                 'HR_sensitivity': sens['HR'], 'HR_sensitivity_lo': sens['lo'],
                 'HR_sensitivity_hi': sens['hi'], 'p_sensitivity': sens['p']})

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(output_dir, 'cox_composite_score.csv'), index=False)
    return summary_df


# =============================
# DOSE-RESPONSE LIKELIHOOD-RATIO TESTS
# =============================

def dose_response_lrt(df, output_dir):
    """Nested LRTs on the stratified (primary) models, from partial
    log-likelihoods. Models share the same rows, so the comparisons are valid."""
    REF_MET = 'metastatic_Non-metastatic'
    REF_SCORE_CAT = 'score_cat_0'

    d_null = build_design_matrix(df, None, 'stratified', [REF_MET])
    d_lin = build_design_matrix(df, 'score_linear', 'stratified', [REF_MET])
    d_cat = build_design_matrix(df, 'score_cat', 'stratified', [REF_MET, REF_SCORE_CAT])

    n = len(d_null)
    if not (len(d_lin) == n and len(d_cat) == n):
        raise ValueError(
            f"Nested models differ in N (null={len(d_null)}, "
            f"linear={len(d_lin)}, categorical={len(d_cat)}); LRT invalid."
        )
    ev = int(d_null['event'].sum())

    ll_null = fit_cox(d_null, 'stratified').log_likelihood_
    ll_lin = fit_cox(d_lin, 'stratified').log_likelihood_
    ll_cat = fit_cox(d_cat, 'stratified').log_likelihood_

    # categorical adds 3 score params (1, 2, 3+); linear adds 1.
    k_cat, k_lin = 3, 1

    def lrt(ll_full, ll_red, dfree):
        stat = 2.0 * (ll_full - ll_red)
        return stat, dfree, float(chi2.sf(stat, dfree))

    tests = []
    s, d_, p = lrt(ll_lin, ll_null, k_lin)
    tests.append({'test': 'Linear trend (ordinal score vs no score)',
                  'chi2': s, 'df': d_, 'p': p})
    s, d_, p = lrt(ll_cat, ll_lin, k_cat - k_lin)
    tests.append({'test': 'Departure from linearity (categorical vs linear)',
                  'chi2': s, 'df': d_, 'p': p})
    s, d_, p = lrt(ll_cat, ll_null, k_cat)
    tests.append({'test': 'Overall score effect (categorical vs no score)',
                  'chi2': s, 'df': d_, 'p': p})

    out = pd.DataFrame(tests)
    out['N'] = n
    out['events'] = ev
    out.to_csv(os.path.join(output_dir, 'dose_response_lrt.csv'), index=False)
    return out


# =============================
# RUN
# =============================

def main(output_dir=RESULTS_DIR):
    out_dir = os.path.join(output_dir, 'composite_score')
    os.makedirs(out_dir, exist_ok=True)

    raw_df = pd.read_csv(RAW_PRISM_CSV, index_col=INDEX,
                         engine='python', encoding='latin1')
    df = prepare_survival_dataframe(raw_df)

    print(f"\n=== Analytic cohort ===")
    print(f"N: {len(df):,}   deaths: {int(df['event'].sum()):,}   "
          f"median follow-up: {df['time_months'].median():.1f} mo")

    med_df, fig_path = kaplan_meier_curves(df, out_dir)
    print(f"\n=== KM median OS by score ===")
    print(med_df.round(1).to_string(index=False))

    summary_df = cox_proportional_hazards(df, out_dir)
    print(f"\n=== Cox HRs (composite score) ===")
    print(summary_df.round(3).to_string(index=False))

    lrt_df = dose_response_lrt(df, out_dir)
    print(f"\n=== Dose-response likelihood-ratio tests ===")
    print(lrt_df.assign(chi2=lambda d: d['chi2'].round(2),
                        p=lambda d: d['p'].map(lambda x: f"{x:.4g}")
                        ).to_string(index=False))

    print(f"\nAll outputs written to: {out_dir}/")
    return summary_df, lrt_df, med_df


if __name__ == '__main__':
    main()


