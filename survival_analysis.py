from preprocesser import preprocess
from lifelines import KaplanMeierFitter, statistics, CoxPHFitter
import matplotlib.pyplot as plt
from globals import SOC_ISO_FEATS
from globals import RESULTS_DIR, PSYCH_DIST_FEATS
from copy import deepcopy
import pandas as pd
import os.path


def plot_KM_curve(df, feat, ax):
    # Select those with True vs False for this social isoluation feature
    df_true = df.loc[df[feat]]
    df_neg = df.loc[~df[feat]]

    # Extract the durations (how long observed/till death) and event observed (whether they died)
    true_durations = df_true['observed_mo']
    true_event_observed = df_true['died']

    false_durations = df_neg['observed_mo']
    false_event_observed = df_neg['died']

    # print(df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()))
    # print(df.dtypes)

    # create kmf objects
    true_kmf = KaplanMeierFitter()
    false_kmf = KaplanMeierFitter()

    # Fit the data into the model
    true_kmf.fit(true_durations, true_event_observed, label=f'{feat} Pts KM Estimate')
    false_kmf.fit(false_durations, false_event_observed, label=f'Not {feat} Pts KM Estimate')

    # Compare using logrank test
    logrankstats = statistics.logrank_test(true_durations, false_durations, true_event_observed,
                                           false_event_observed, t_0=84)
    logrank_p_value = logrankstats.p_value

    # Create a Plot
    print(f"plotting KM Survival Curve for {feat} ")
    ax.set_title(f'KM Survival Curve'
                 '\n'
                 f'{len(df_true.index)} Pts w/ {feat}  vs {len(df_neg.index)} Pts w/o {feat}'
                 '\n'
                 # f'psyhx subset = {psyhx_subset}, help subset = {help_subset}'
                 # '\n'
                 f'Logrank Chi^2 p-value: {logrank_p_value:.4f}')

    true_kmf.plot_survival_function(ax=ax)
    false_kmf.plot_survival_function(ax=ax)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)


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

    # Make df of pts who see and do not see disiplcine
    # SOC_ISO_FEATS = ['NoAssistance', 'Social_isolation_YN'] # Use if you want to keep noassistance alone
    for feat in SOC_ISO_FEATS + PSYCH_DIST_FEATS:
        # Create a MatPlotLib figure with two subplots, add the plots, and save
        fig, axs = plt.subplots(1)
        fig.set_figheight(15)
        fig.set_figwidth(15)

        plot_KM_curve(df, feat, axs)
        # plot_CPH_model(df_dspln, dspln_to_link, axs[1])
        fig_filename = results_dir + 'survival_' + feat + ".png"
        fig.savefig(fig_filename)
        print(f"printed to {fig_filename}")
        plt.show()
        plt.close(fig)

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


if __name__ == '__main__':
    survival_analysis()
    print("analysis complete!")
