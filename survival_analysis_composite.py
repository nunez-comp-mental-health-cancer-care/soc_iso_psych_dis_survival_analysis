"""
Survival analysis for composite social isolation score.
For JCO Oncology Practice resubmission - addresses reviewer comment on dose-response.

(1) Kaplan-Meier survival curves stratified by social isolation score (0, 1, 2, 3+)
    Figure styling matches existing manuscript KM figures:
      - No CI shading
      - No log-rank p-value annotation (per prior reviewer comment)
      - Median OS labels with 95% CI in bottom-right box
      - At-risk / Censored / Events tables beneath x-axis
      - X-axis: "Time from initial oncology visit (months)"

(2) Cox PH models stratified by tumour group:
    - Score as categorical (reference = 0)
    - Score as linear ordinal (per 1-unit HR)
    - Likelihood ratio test for linearity (categorical vs linear)

Treat-as-negative scoring is used throughout (consistent with the rest of the
manuscript and with prior analyses of this cohort). Patients with all five
domains unavailable (n = 717) are excluded.

Dependencies: pip install lifelines matplotlib pandas numpy scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
from lifelines.utils import median_survival_times
from lifelines.plotting import add_at_risk_counts

from globals import RAW_PRISM_CSV, DATA_DIR, INDEX


# BC Cancer TumourGroup codes -> display category
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


def add_social_isolation_score(raw_df):
    """Add composite social isolation score (0-5) using treat-as-negative
    scoring for missing/unknown responses (consistent with the rest of the
    manuscript). Also adds a count of observed (non-missing) domains so
    patients with all five domains unavailable can be excluded."""
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


def prepare_survival_dataframe(raw_df):
    """Build the analysis DataFrame for survival analyses."""
    df = raw_df.copy()
    df = add_social_isolation_score(df)

    # Time and event
    df['time_months'] = pd.to_numeric(df['Dx_to_death_lastfu_months'], errors='coerce')
    df['event'] = (df['PatStatus'].astype(str).str.strip() == 'D').astype(int)

    # Score collapsed to 0, 1, 2, 3+ for KM and categorical Cox.
    # Use '3plus' rather than '3+' to avoid issues with '+' in column names.
    df['score_cat'] = df['social_isolation_score'].apply(
        lambda x: '3plus' if x >= 3 else str(int(x))
    )
    df['score_linear'] = df['social_isolation_score'].astype(int)

    # Confounders
    df['age'] = pd.to_numeric(df['AgeAtDiagnosis'], errors='coerce')
    df['female'] = (df['Sex'].astype(str).str.strip() == 'F').astype(int)

    # Tumour group (BC Cancer service-line classification, used as Cox strata)
    tg = df['TumourGroup'].fillna('').astype(str).str.strip().str.upper()
    df['tumour_group'] = tg.map(lambda v: TUMOUR_GROUP_MAP.get(v, 'Other / Unknown'))

    # Performance status: keep unknown (88, 99, blank) as a category
    ps_numeric = pd.to_numeric(df['PerformanceStatus'], errors='coerce')
    df['performance_status'] = ps_numeric.where(
        ~(ps_numeric.isna() | ps_numeric.isin([88, 99])),
        other=-1
    ).astype(int).astype(str)
    df['performance_status'] = df['performance_status'].replace('-1', 'Unknown')

    # Metastatic status at diagnosis
    met = df['M1_IV_ALL_CAT'].astype(str).str.strip()
    df['metastatic'] = met.map(
        {'1': 'Metastatic', '0': 'Non-metastatic', '99': 'Unknown'}
    ).fillna('Unknown')

    # Exclude patients with no observed social isolation domains
    df = df[df['social_isolation_n_observed'] > 0].copy()
    # Defensive: drop any rows missing time/event/age
    df = df.dropna(subset=['time_months', 'event', 'age']).copy()
    df = df[df['time_months'] >= 0].copy()
    return df


def kaplan_meier_analysis(df, output_dir=DATA_DIR):
    """KM curves stratified by social isolation score (0, 1, 2, 3+),
    styled to match existing manuscript figures."""
    score_display_map = {'0': '0', '1': '1', '2': '2', '3plus': '3+'}
    score_categories = ['0', '1', '2', '3plus']
    # Matplotlib default cycle: blue/orange of existing figures, extended as
    # cool-to-warm gradient for ordinal dose-response.
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(10, 7))
    kmf_results = {}

    for cat, color in zip(score_categories, colors):
        mask = df['score_cat'] == cat
        kmf = KaplanMeierFitter()
        kmf.fit(
            df.loc[mask, 'time_months'], df.loc[mask, 'event'],
            label=f'Score {score_display_map[cat]}'
        )
        kmf.plot_survival_function(ax=ax, color=color, ci_show=False)
        kmf_results[cat] = kmf

    # Log-rank computed for console only (not shown on figure per prior reviewer)
    lr = multivariate_logrank_test(df['time_months'], df['score_cat'], df['event'])

    # Median OS box: bottom-right, square corners, white fill, thin black border
    median_lines = []
    for cat in score_categories:
        kmf = kmf_results[cat]
        median = kmf.median_survival_time_
        try:
            ci_df = median_survival_times(kmf.confidence_interval_)
            ci_vals = sorted([ci_df.iloc[0, 0], ci_df.iloc[0, 1]])
            lo, hi = ci_vals[0], ci_vals[1]
            ci_df = median_survival_times(kmf.confidence_interval_)
            if np.isinf(median) or np.isinf(lo) or np.isinf(hi):
                median_lines.append(f"Score {score_display_map[cat]}: Not reached")
            else:
                median_lines.append(
                    f"Score {score_display_map[cat]}: {median:.1f} mo "
                    f"(95% CI {lo:.1f}-{hi:.1f})"
                )
        except Exception:
            median_lines.append(f"Score {score_display_map[cat]}: see table")

    ax.text(
        0.02, 0.02, "\n".join(median_lines),
        transform=ax.transAxes, fontsize=8, va='bottom', ha='left',
        bbox=dict(
            boxstyle='square,pad=0.4',
            facecolor='white',
            edgecolor='black',
            linewidth=1.0,
        ),
    )

    ax.set_xlabel('Time from initial oncology visit (months)', fontsize=11)
    ax.set_ylabel('Proportion alive', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 100)
    ax.legend(loc='upper right', fontsize=10, frameon=True)

    # At-risk / censored / events tables below the plot
    add_at_risk_counts(*kmf_results.values(), ax=ax,
                       rows_to_show=['At risk', 'Censored', 'Events'])

    plt.tight_layout()
    plt.savefig(f'{output_dir}km_social_isolation_score.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}km_social_isolation_score.pdf',
                bbox_inches='tight')
    plt.close()

    # Console summary
    print("\n=== Kaplan-Meier results: Social isolation score ===")
    header = f"{'Score':<8}{'N':>10}{'Deaths':>10}{'Median OS (mo)':>20}{'95% CI':>30}"
    print(header)
    print('-' * len(header))
    for cat in score_categories:
        kmf = kmf_results[cat]
        median = kmf.median_survival_time_
        try:
            ci_df = median_survival_times(kmf.confidence_interval_)
            lo, hi = ci_df.iloc[0, 0], ci_df.iloc[0, 1]
            ci_str = f"{lo:.1f} - {hi:.1f}" if not (np.isinf(lo) or np.isinf(hi)) else "Not reached"
        except Exception:
            ci_str = "see plot"
        n = int((df['score_cat'] == cat).sum())
        events = int(df.loc[df['score_cat'] == cat, 'event'].sum())
        median_str = f"{median:.1f}" if not np.isinf(median) else "Not reached"
        print(f"{score_display_map[cat]:<8}{n:>10,}{events:>10,}{median_str:>20}{ci_str:>30}")

    print(f"\nOverall log-rank test (console only, NOT shown on figure): "
          f"χ² = {lr.test_statistic:.3f}, p = {lr.p_value:.4g}")
    print("Use Cox-adjusted p-values for manuscript text per prior reviewer comment.")

    print("\n=== Pairwise log-rank tests (vs Score 0; console only) ===")
    for cat in ['1', '2', '3plus']:
        ref = df['score_cat'] == '0'
        cmp = df['score_cat'] == cat
        pair = logrank_test(
            df.loc[ref, 'time_months'], df.loc[cmp, 'time_months'],
            df.loc[ref, 'event'], df.loc[cmp, 'event']
        )
        print(f"  0 vs {score_display_map[cat]}: χ² = {pair.test_statistic:.3f}, p = {pair.p_value:.4g}")
    return kmf_results, lr


def build_design_matrix(model_df, score_col, ref_levels, min_count=20):
    """One-hot encode categorical confounders, drop reference levels, drop
    sparse dummies to avoid singular-matrix issues in Cox fitting. Tumour
    group is kept as a string column (used for stratification in Cox)."""
    keep = ['time_months', 'event', score_col, 'age', 'female',
            'performance_status', 'metastatic', 'tumour_group']
    sub = model_df[keep].copy()

    sub = pd.get_dummies(
        sub, columns=['performance_status', 'metastatic'],
        drop_first=False, dtype=int
    )
    if score_col == 'score_cat':
        sub = pd.get_dummies(sub, columns=['score_cat'], drop_first=False, dtype=int)

    for ref_col in ref_levels:
        if ref_col in sub.columns:
            sub = sub.drop(columns=[ref_col])

    # Drop sparse dummies. Never drop time, event, score predictors, or strata.
    protected = {'time_months', 'event', 'age', 'female', 'tumour_group', score_col}
    dropped = []
    for c in list(sub.columns):
        if c in protected or c.startswith('score_'):
            continue
        col_sum = sub[c].sum()
        if col_sum < min_count:
            dropped.append((c, int(col_sum)))
            sub = sub.drop(columns=[c])
    if dropped:
        print(f"  Dropped sparse dummy columns (n < {min_count}):")
        for col, n in dropped:
            print(f"    {col}: n={n}")

    numeric_cols = [c for c in sub.columns if c != 'tumour_group']
    for c in numeric_cols:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')
    sub = sub.dropna(subset=numeric_cols)
    return sub


def cox_proportional_hazards(df, output_dir=DATA_DIR):
    """Cox PH stratified by tumour group (separate baseline hazard per group)."""
    REF_PS = 'performance_status_0'
    REF_MET = 'metastatic_Non-metastatic'
    REF_SCORE_CAT = 'score_cat_0'
    common_refs = [REF_PS, REF_MET]

    # ----- Categorical model -----
    print("\n--- Building design matrix: categorical model ---")
    cat_df = build_design_matrix(
        df, score_col='score_cat',
        ref_levels=common_refs + [REF_SCORE_CAT]
    )
    cph_cat = CoxPHFitter()
    cph_cat.fit(cat_df, duration_col='time_months', event_col='event',
                strata=['tumour_group'], show_progress=False)

    print("\n=== Cox PH (strata: tumour group): score CATEGORICAL (ref = 0) ===")
    print(cph_cat.summary[['coef', 'exp(coef)',
                           'exp(coef) lower 95%', 'exp(coef) upper 95%',
                           'p']].round(4))
    cph_cat.summary.to_csv(f'{output_dir}cox_social_isolation_categorical.csv')

    # ----- Linear ordinal model -----
    print("\n--- Building design matrix: linear ordinal model ---")
    lin_df = build_design_matrix(
        df, score_col='score_linear', ref_levels=common_refs
    )
    cph_lin = CoxPHFitter()
    cph_lin.fit(lin_df, duration_col='time_months', event_col='event',
                strata=['tumour_group'], show_progress=False)

    print("\n=== Cox PH (strata: tumour group): score LINEAR ORDINAL (per +1) ===")
    print(cph_lin.summary[['coef', 'exp(coef)',
                           'exp(coef) lower 95%', 'exp(coef) upper 95%',
                           'p']].round(4))
    cph_lin.summary.to_csv(f'{output_dir}cox_social_isolation_linear.csv')

    # ----- Likelihood ratio test for linearity -----
    df_diff = 3 - 1
    lr_stat = 2 * (cph_cat.log_likelihood_ - cph_lin.log_likelihood_)
    lr_p = 1 - chi2.cdf(lr_stat, df_diff)
    print("\n=== Likelihood ratio test: linearity of dose-response ===")
    print(f"  -2(LL_lin - LL_cat) = {lr_stat:.3f}  (df = {df_diff})")
    print(f"  p = {lr_p:.4g}")
    if lr_p > 0.05:
        print("  → Linearity is reasonable. Linear HR can be reported as primary.")
    else:
        print("  → Linearity rejected. Report categorical HRs as primary.")

    # ----- Proportional hazards assumption check -----
    print("\n=== Schoenfeld residuals: PH assumption check (categorical model) ===")
    try:
        cph_cat.check_assumptions(cat_df, p_value_threshold=0.05, show_plots=False)
    except Exception as e:
        print(f"  PH check raised an exception: {e}")

    return cph_cat, cph_lin, {'lr_stat': lr_stat, 'lr_p': lr_p, 'df': df_diff}


if __name__ == '__main__':
    raw_df = pd.read_csv(RAW_PRISM_CSV, index_col=INDEX,
                         engine='python', encoding='latin1')
    df = prepare_survival_dataframe(raw_df)

    print(f"\n=== Analytic cohort ===")
    print(f"N total:            {len(df):,}")
    print(f"Total deaths:       {int(df['event'].sum()):,}")
    print(f"Median follow-up:   {df['time_months'].median():.1f} months")

    print(f"\nDistribution of collapsed social isolation score:")
    display_map = {'0': '0', '1': '1', '2': '2', '3plus': '3+'}
    counts = df['score_cat'].value_counts().reindex(['0', '1', '2', '3plus'])
    pcts = (counts / len(df) * 100).round(2)
    for cat in ['0', '1', '2', '3plus']:
        print(f"  {display_map[cat]}: {counts[cat]:>7,} ({pcts[cat]:>5.2f}%)")

    print(f"\nPerformance status:")
    print(df['performance_status'].value_counts().sort_index().to_string())
    print(f"\nMetastatic status:")
    print(df['metastatic'].value_counts().to_string())
    print(f"\nTumour group (used as Cox strata):")
    print(df['tumour_group'].value_counts().to_string())

    kmf_results, lr_test = kaplan_meier_analysis(df)
    cph_cat, cph_lin, lr_linearity = cox_proportional_hazards(df)

    print("\nAnalysis complete.")
    print(f"Outputs in {DATA_DIR}:")
    print("  km_social_isolation_score.png / .pdf")
    print("  cox_social_isolation_categorical.csv")
    print("  cox_social_isolation_linear.csv")
    
