from preprocesser import preprocess
from lifelines import KaplanMeierFitter, statistics, CoxPHFitter
import matplotlib.pyplot as plt
from globals import SOC_ISO_FEATS
from globals import RESULTS_DIR, PSYCH_DIST_FEATS
from copy import deepcopy
import pandas as pd
import os.path

PAPER_LABELS = {
    'LiveAlone': ('Living alone', 'Living with others'),
    'LostSpousePartner': ('Recent partner loss', 'No recent partner loss'),
    'NoRegularContact': ('No regular social contact', 'Regular social contact'),
    'NoEmotionalSupport': ('No emotional support', 'Has emotional support'),
    'NoAssistance': ('No assistance with IADLs', 'Has assistance with IADLs'),
    'Social_isolation_YN': ('Social isolation', 'No social isolation'),
    'AnxietyScore_CAT2': ('Anxiety', 'No anxiety'),
    'DepressionScore_CAT2': ('Depression', 'No depression'),
}


def km_analysis_stats(df, feat, t_0=84):
    # Select those with True vs False for this social isoluation feature
    feat_df = df[['observed_mo', 'died', feat]].copy()
    feat_df['observed_mo'] = pd.to_numeric(feat_df['observed_mo'], errors='coerce')
    feat_df['died'] = feat_df['died'].astype(int)
    feat_df = feat_df.dropna()
    df_true = feat_df.loc[feat_df[feat]]
    df_neg = feat_df.loc[~feat_df[feat]]

    # Extract the durations (how long observed/till death) and event observed (whether they died)
    true_durations = df_true['observed_mo']
    true_event_observed = df_true['died']

    false_durations = df_neg['observed_mo']
    false_event_observed = df_neg['died']

    # create kmf objects
    true_kmf = KaplanMeierFitter()
    false_kmf = KaplanMeierFitter()

    # Fit the data into the model
    true_kmf.fit(true_durations, true_event_observed, label=f'{feat} Pts KM Estimate')
    false_kmf.fit(false_durations, false_event_observed, label=f'Not {feat} Pts KM Estimate')

    # Compare using logrank test
    logrankstats = statistics.logrank_test(true_durations, false_durations, true_event_observed,
                                           false_event_observed, t_0=t_0)

    return {
        'feature': feat,
        'true_kmf': true_kmf,
        'false_kmf': false_kmf,
        'df_true': df_true,
        'df_false': df_neg,
        'logrank_p_value': logrankstats.p_value,
        'logrank_test_statistic': logrankstats.test_statistic,
        't_0_months': t_0,
        'n_true': len(df_true),
        'n_false': len(df_neg),
        'events_true': int(true_event_observed.sum()),
        'events_false': int(false_event_observed.sum()),
        'median_survival_true': true_kmf.median_survival_time_,
        'median_survival_false': false_kmf.median_survival_time_,
    }


def plot_KM_curve(df, feat, ax, paper_friendly=False, t_0=84):
    stats = km_analysis_stats(df, feat, t_0=t_0)
    true_kmf = stats['true_kmf']
    false_kmf = stats['false_kmf']
    logrank_p_value = stats['logrank_p_value']

    # Create a Plot
    print(f"plotting KM Survival Curve for {feat} ")
    if paper_friendly:
        pos_label, neg_label = PAPER_LABELS.get(feat, (feat, f'Not {feat}'))
        paper_fontsize = 14
        true_kmf.plot_survival_function(ax=ax, label=pos_label, ci_show=False, linewidth=2.5)
        false_kmf.plot_survival_function(ax=ax, label=neg_label, ci_show=False, linewidth=2.5)
        ax.set_title(f'Survival Functions: {pos_label}', fontsize=paper_fontsize, fontweight='bold')
        ax.set_xlabel('Time from initial oncology visit (months)', fontsize=paper_fontsize)
        ax.set_ylabel('Proportion alive', fontsize=paper_fontsize)
        ax.tick_params(axis='both', labelsize=paper_fontsize)
        ax.legend(fontsize=paper_fontsize)
        ax.grid(False)
    else:
        ax.set_title(f'KM Survival Curve'
                     '\n'
                     f'{stats["n_true"]} Pts w/ {feat}  vs {stats["n_false"]} Pts w/o {feat}'
                     '\n'
                     f'Logrank Chi^2 p-value: {logrank_p_value:.4f}')
        true_kmf.plot_survival_function(ax=ax)
        false_kmf.plot_survival_function(ax=ax)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    return stats


def plot_CPH_model(df, ax, feats):
    """
    Fits a Cox Proportional Hazard model to the survival of pts who saw a specific disipline e.g. psychiatry

    Examines the possible imapct of features in the CPH_COLS global

    :param df_dspln: df of pts who saw a specific disicpline
    :param dspln_to_link: discipline to conduct analysis on
    :param ax: matplotlib subplot axis to plot onto
    :param name: string to name this cph model
    :param feats: features to use for the CPH model, in addition to metastatic status, age and gender
    :return: nothing, but plots figure to supplied axis
    """
    # Make a df from the discipline's df which just the columns we need or are examining
    cph_cols = ['age_dx', 'observed_mo', 'male', 'died',
                'M1_IV_ALL_CAT'] + feats  # Columns for our CPH Model, CPH_COLS are what variables are examined
    # cph_cols = ['observed_mo', 'died'] + feats # Use to keep by itself
    df = df.copy()[cph_cols]
    bool_cols = ['male', 'died', 'M1_IV_ALL_CAT'] + feats
    df[bool_cols] = df[bool_cols].astype(int)
    df['observed_mo'] = pd.to_numeric(df['observed_mo'], errors='coerce')
    df['age_dx'] = pd.to_numeric(df['age_dx'], errors='coerce')
    df = df.dropna()
    print(f"CPH analysis includes {len(df)} patients")

    # Prepare plot
    ax.set_title(f'Cox Proportional Hazard Coefficeints for Mortality')

    # Fit CoxPHFitter
    cph = CoxPHFitter()  # Instantiate the class to create a cph object
    cph.fit(df, 'observed_mo', event_col='died')
    cph.print_summary()

    print(f"plotting CPH Ratios")
    cph.plot(ax=ax)

    return cph.summary


def survival_analysis(results_dir=RESULTS_DIR):
    # Use preprocessor to make df with our pt data
    df = preprocess()
    km_summary_rows = []
    paper_dir = os.path.join(results_dir, 'paper_figures')
    os.makedirs(paper_dir, exist_ok=True)

    # Make df of pts who see and do not see disiplcine
    # SOC_ISO_FEATS = ['NoAssistance', 'Social_isolation_YN'] # Use if you want to keep noassistance alone
    for feat in SOC_ISO_FEATS + PSYCH_DIST_FEATS:
        # Create a MatPlotLib figure with two subplots, add the plots, and save
        fig, axs = plt.subplots(1)
        fig.set_figheight(15)
        fig.set_figwidth(15)

        stats = plot_KM_curve(df, feat, axs)
        km_summary_rows.append({
            'feature': feat,
            'n_true': stats['n_true'],
            'n_false': stats['n_false'],
            'events_true': stats['events_true'],
            'events_false': stats['events_false'],
            'median_survival_true_mo': stats['median_survival_true'],
            'median_survival_false_mo': stats['median_survival_false'],
            'logrank_p_value': stats['logrank_p_value'],
            'logrank_test_statistic': stats['logrank_test_statistic'],
            'logrank_t_0_months': stats['t_0_months'],
        })
        # plot_CPH_model(df_dspln, dspln_to_link, axs[1])
        fig_filename = results_dir + 'survival_' + feat + ".png"
        fig.savefig(fig_filename)
        print(f"printed to {fig_filename}")
        plt.show()
        plt.close(fig)

        paper_fig, paper_ax = plt.subplots(1, figsize=(8, 8))
        plot_KM_curve(df, feat, paper_ax, paper_friendly=True)
        paper_filename = os.path.join(paper_dir, f'paper_survival_{feat}.png')
        paper_fig.tight_layout()
        paper_fig.savefig(paper_filename, dpi=300)
        print(f"printed to {paper_filename}")
        plt.close(paper_fig)

    # Plot the CPHs for both papers
    soc_iso_feats_no_any = deepcopy(SOC_ISO_FEATS)
    soc_iso_feats_no_any.remove('Social_isolation_YN')  # Don't include "Social_isolation_YN" for CPH model

    for name, feats in zip(['soc_iso', 'psych_dist'], [soc_iso_feats_no_any, PSYCH_DIST_FEATS]):
        # For social isolation CPH, create a matplot figure with two subplots, add the plots, and save
        fig, axs = plt.subplots(1)
        fig.set_figheight(15)
        fig.set_figwidth(15)

        cph_summary = plot_CPH_model(df, axs, feats)
        cph_filename = os.path.join(results_dir, name + '_cph.csv')
        cph_summary.to_csv(cph_filename)

        fig_filename = results_dir + name + '_cph.png'
        fig.savefig(fig_filename)
        print(f"printed to {fig_filename}")
        plt.close(fig)

    km_summary_path = os.path.join(results_dir, 'km_summary.csv')
    pd.DataFrame(km_summary_rows).to_csv(km_summary_path, index=False)
    print(f"printed to {km_summary_path}")


if __name__ == '__main__':
    survival_analysis()
    print("analysis complete!")
