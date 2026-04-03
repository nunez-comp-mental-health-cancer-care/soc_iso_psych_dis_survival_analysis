from contextlib import redirect_stdout
from copy import deepcopy
import io
import os

import pandas as pd
from lifelines import CoxPHFitter

from globals import PSYCH_DIST_FEATS, RESULTS_DIR, SOC_ISO_FEATS
from preprocesser import preprocess
from survival_analysis_combined import CPH_COLS as COMBINED_CPH_COLS
from survival_analysis_combined import FEAT_RENAME


RAW_COLUMN_MAP = {
    'died': 'PatStatus',
    'male': 'Sex',
    'M1_IV_ALL_CAT': 'M1_IV_ALL_CAT',
    'LiveAlone': 'LiveAlone',
    'LostSpousePartner': 'LostSpousePartner',
    'NoAssistance': 'Assistance',
    'NoRegularContact': 'RegularContact',
    'NoEmotionalSupport': 'EmotionalSupport',
    'Social_isolation_YN': 'Social_isolation_YN',
    'AnxietyScore_CAT2': 'AnxietyScore_CAT2',
    'DepressionScore_CAT2': 'DepressionScore_CAT2',
    'female': 'Sex',
    'metastatic': 'M1_IV_ALL_CAT',
    'lives_alone': 'LiveAlone',
    'lost_partner': 'LostSpousePartner',
    'no_iadl_assistance': 'Assistance',
    'no_regular_contact': 'RegularContact',
    'no_emotional_support': 'EmotionalSupport',
    'anxiety': 'AnxietyScore_CAT2',
    'depression': 'DepressionScore_CAT2',
}


def load_qc_summary():
    qc_path = os.path.join('data', 'preprocess_qc_summary.csv')
    qc_df = pd.read_csv(qc_path)
    return qc_df.set_index('column')


def prepare_grouped_cph_df(df, feats):
    cph_cols = ['age_dx', 'observed_mo', 'male', 'died', 'M1_IV_ALL_CAT'] + feats
    model_df = df.copy()[cph_cols]
    bool_cols = ['male', 'died', 'M1_IV_ALL_CAT'] + feats
    model_df[bool_cols] = model_df[bool_cols].astype(int)
    model_df['observed_mo'] = pd.to_numeric(model_df['observed_mo'], errors='coerce')
    model_df['age_dx'] = pd.to_numeric(model_df['age_dx'], errors='coerce')
    model_df = model_df.dropna()
    return model_df


def prepare_combined_cph_df(df):
    model_df = df.copy()[COMBINED_CPH_COLS]
    model_df['male'] = ~model_df['male']
    bool_cols = [col for col in COMBINED_CPH_COLS if col != 'observed_mo']
    model_df[bool_cols] = model_df[bool_cols].astype(int)
    model_df['observed_mo'] = pd.to_numeric(model_df['observed_mo'], errors='coerce')
    model_df = model_df.dropna()
    model_df = model_df.rename(columns=FEAT_RENAME)
    return model_df


def model_manifest_row(model_name, model_df):
    covariates = [col for col in model_df.columns if col not in ['observed_mo', 'died']]
    n_events = int(model_df['died'].sum())
    n_covariates = len(covariates)
    epv = round(n_events / n_covariates, 2) if n_covariates else None
    return {
        'model_name': model_name,
        'n_observations': len(model_df),
        'n_events': n_events,
        'n_covariates': n_covariates,
        'events_per_variable': epv,
        'covariates': '|'.join(covariates),
    }


def save_collinearity_outputs(model_name, model_df, output_dir):
    covariates = [col for col in model_df.columns if col not in ['observed_mo', 'died']]
    corr = model_df[covariates].corr()
    corr_path = os.path.join(output_dir, f'{model_name}_correlation_matrix.csv')
    corr.to_csv(corr_path)

    rows = []
    for i, col_a in enumerate(covariates):
        for col_b in covariates[i + 1:]:
            corr_val = corr.loc[col_a, col_b]
            rows.append({
                'variable_a': col_a,
                'variable_b': col_b,
                'correlation': corr_val,
                'abs_correlation': abs(corr_val),
                'flag_abs_corr_ge_0_7': abs(corr_val) >= 0.7,
            })

    pairwise_df = pd.DataFrame(rows).sort_values('abs_correlation', ascending=False)
    pairwise_path = os.path.join(output_dir, f'{model_name}_pairwise_correlations.csv')
    pairwise_df.to_csv(pairwise_path, index=False)
    return corr_path, pairwise_path


def save_imputation_summary(model_name, covariates, qc_summary, output_dir):
    rows = []
    for covariate in covariates:
        raw_column = RAW_COLUMN_MAP.get(covariate)
        if raw_column is None or raw_column not in qc_summary.index:
            rows.append({
                'model_covariate': covariate,
                'raw_column': raw_column,
                'true_count': None,
                'false_count': None,
                'imputed_count': None,
                'unmapped_count': None,
            })
            continue

        qc_row = qc_summary.loc[raw_column]
        rows.append({
            'model_covariate': covariate,
            'raw_column': raw_column,
            'true_count': int(qc_row.get('true', 0)),
            'false_count': int(qc_row.get('false', 0)),
            'imputed_count': int(qc_row.get('imputed', 0)),
            'unmapped_count': int(qc_row.get('unmapped', 0)),
        })

    path = os.path.join(output_dir, f'{model_name}_imputation_summary.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_cph_assumptions(model_name, model_df, output_dir, qc_summary):
    cph = CoxPHFitter()
    cph.fit(model_df, 'observed_mo', event_col='died')

    summary_path = os.path.join(output_dir, f'{model_name}_cph_summary.csv')
    cph.summary.to_csv(summary_path)

    assumptions_path = os.path.join(output_dir, f'{model_name}_assumptions.txt')
    captured = io.StringIO()
    with redirect_stdout(captured):
        cph.check_assumptions(model_df, p_value_threshold=0.05, show_plots=False)

    with open(assumptions_path, 'w', encoding='utf-8') as f:
        f.write(captured.getvalue())

    corr_path, pairwise_path = save_collinearity_outputs(model_name, model_df, output_dir)
    covariates = [col for col in model_df.columns if col not in ['observed_mo', 'died']]
    imputation_path = save_imputation_summary(model_name, covariates, qc_summary, output_dir)

    print(f"Saved {model_name} Cox summary to {summary_path}")
    print(f"Saved {model_name} assumptions check to {assumptions_path}")
    print(f"Saved {model_name} correlation matrix to {corr_path}")
    print(f"Saved {model_name} pairwise correlations to {pairwise_path}")
    print(f"Saved {model_name} imputation summary to {imputation_path}")
    return model_manifest_row(model_name, model_df)


def km_diagnostics(df):
    rows = []
    for feat in SOC_ISO_FEATS + PSYCH_DIST_FEATS:
        feat_df = df[['observed_mo', 'died', feat]].copy()
        feat_df['observed_mo'] = pd.to_numeric(feat_df['observed_mo'], errors='coerce')
        feat_df['died'] = feat_df['died'].astype(int)
        feat_df = feat_df.dropna()

        df_true = feat_df.loc[feat_df[feat]]
        df_false = feat_df.loc[~feat_df[feat]]
        rows.append({
            'feature': feat,
            'n_true': len(df_true),
            'n_false': len(df_false),
            'events_true': int(df_true['died'].sum()),
            'events_false': int(df_false['died'].sum()),
            'event_rate_true': round(df_true['died'].mean(), 4),
            'event_rate_false': round(df_false['died'].mean(), 4),
            'median_observed_mo_true': round(df_true['observed_mo'].median(), 2),
            'median_observed_mo_false': round(df_false['observed_mo'].median(), 2),
        })
    return pd.DataFrame(rows)


def write_assumptions_overview(output_dir):
    overview_path = os.path.join(output_dir, 'assumptions_overview.txt')
    content = """Checks included in this folder:
- Cox proportional hazards diagnostics via lifelines check_assumptions for soc_iso, psych_dist, and combined models.
- Events-per-variable summaries for each Cox model.
- Pairwise correlation screens for modeled covariates.
- Model-level imputation burden summaries derived from preprocess_qc_summary.csv.
- Kaplan-Meier group diagnostics for each binary feature used in survival_analysis.py.

Interpretation notes:
- The proportional hazards test is sensitive with large sample sizes, so small deviations may be flagged.
- Pairwise correlation is a screen for collinearity risk, not a full causal or design-based assessment.
- Imputation summaries reflect preprocessing rules, not necessarily missingness after any future code changes.

Not fully testable in this script:
- Non-informative censoring.
- Independence of observations.
- Correct functional form for continuous predictors beyond generic lifelines guidance.
- Clinical validity of the preprocessing conventions themselves.
"""
    with open(overview_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return overview_path


def run_assumptions_checks(results_dir=RESULTS_DIR):
    output_dir = os.path.join(results_dir, 'assumptions_checks')
    os.makedirs(output_dir, exist_ok=True)

    df = preprocess()
    qc_summary = load_qc_summary()

    soc_iso_feats = deepcopy(SOC_ISO_FEATS)
    soc_iso_feats.remove('Social_isolation_YN')

    manifests = []
    manifests.append(
        run_cph_assumptions('soc_iso', prepare_grouped_cph_df(df, soc_iso_feats), output_dir, qc_summary)
    )
    manifests.append(
        run_cph_assumptions('psych_dist', prepare_grouped_cph_df(df, PSYCH_DIST_FEATS), output_dir, qc_summary)
    )
    manifests.append(
        run_cph_assumptions('combined', prepare_combined_cph_df(df), output_dir, qc_summary)
    )

    manifest_path = os.path.join(output_dir, 'assumptions_manifest.csv')
    pd.DataFrame(manifests).to_csv(manifest_path, index=False)
    print(f"Saved assumptions manifest to {manifest_path}")

    km_path = os.path.join(output_dir, 'km_diagnostics.csv')
    km_diagnostics(df).to_csv(km_path, index=False)
    print(f"Saved KM diagnostics to {km_path}")

    overview_path = write_assumptions_overview(output_dir)
    print(f"Saved assumptions overview to {overview_path}")


if __name__ == '__main__':
    run_assumptions_checks()
    print("Assumptions checks complete!")
