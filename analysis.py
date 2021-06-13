from preprocesser import preprocess
from lifelines import KaplanMeierFitter, statistics, CoxPHFitter
import matplotlib.pyplot as plt
from globals import SOC_ISO_FEATS, RESULTS_DIR
import pandas as pd



def plot_KM_curve(df, feat, ax):

    ## Select those with True vs False for this social isoluation feature
    df_true = df.loc[df[feat]]
    df_neg = df.loc[~df[feat]]

    ## Extract the durations (how long observed/till death) and event observed (whether they died)
    true_durations = df_true['observed_mo']
    true_event_observed = df_true['died']

    false_durations = df_neg['observed_mo']
    false_event_observed = df_neg['died']

    #print(df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()))
    #print(df.dtypes)



    ## create kmf objects
    true_kmf = KaplanMeierFitter()
    false_kmf = KaplanMeierFitter()

    ## Fit the data into the model
    true_kmf.fit(true_durations, true_event_observed, label=f'{feat} Pts KM Estimate')
    false_kmf.fit(false_durations, false_event_observed, label=f'Not {feat} Pts KM Estimate')

    ## Compare using logrank test
    logrankstats = statistics.logrank_test(true_durations, false_durations, true_event_observed,
                                           false_event_observed, t_0=84)
    logrank_p_value = logrankstats.p_value

    ## Create a Plot
    print(f"plotting KM Survival Curve for {feat} ")
    ax.set_title(f'KM Survival Curve'
                 '\n'
                 f'{len(df_true.index)} Pts w/ {feat}  vs {len(df_neg.index)} Pts w/o {feat}'
                 '\n'
                 #f'psyhx subset = {psyhx_subset}, help subset = {help_subset}'
                 #'\n'
                 f'Logrank Chi^2 p-value: {logrank_p_value:.4f}')

    true_kmf.plot_survival_function(ax=ax)
    false_kmf.plot_survival_function(ax=ax)
    ax.set_xlim(0, 100)


def analysis(results_dir=RESULTS_DIR):
    # Use preprocessor to make df with our pt data
    df = preprocess()

    # Make df of pts who see and do not see disiplcine

    for feat in SOC_ISO_FEATS:
        # Create a matplot figure with two subplots, add the plots, and save
        fig, axs = plt.subplots(1)
        fig.set_figheight(15)
        fig.set_figwidth(15)

        plot_KM_curve(df, feat, axs)
        #plot_CPH_model(df_dspln, dspln_to_link, axs[1])
        fig_filename = results_dir + 'survival_' + feat + ".png"
        fig.savefig(fig_filename)
        print(f"printed to {fig_filename}")
        plt.close(fig)


if __name__ == '__main__':
    analysis()
    print("analysis complete!")