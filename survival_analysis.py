from preprocesser import preprocess
from lifelines import KaplanMeierFitter, statistics
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
from globals import SOC_ISO_FEATS, RESULTS_DIR, PSYCH_DIST_FEATS
import pandas as pd
import os

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

# ✅ FORMAT
def fmt(val):
    if val is None or pd.isna(val):
        return "NR"
    return f"{val:.1f}"

# ✅ MEDIAN + CI
def compute_median_ci(kmf):
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_

    surv = sf.iloc[:, 0]
    lower_ci = ci.iloc[:, 0]
    upper_ci = ci.iloc[:, 1]

    def first_cross(series):
        crossed = series[series <= 0.5]
        return crossed.index[0] if len(crossed) > 0 else None

    median = first_cross(surv)
    lower = first_cross(upper_ci)
    upper = first_cross(lower_ci)

    return median, lower, upper

# ✅ KM STATS
def km_analysis_stats(df, feat):
    temp = df[['observed_mo', 'died', feat]].copy()
    temp['observed_mo'] = pd.to_numeric(temp['observed_mo'], errors='coerce')
    temp['died'] = temp['died'].astype(int)
    temp = temp.dropna()

    df_t = temp[temp[feat] == True]
    df_f = temp[temp[feat] == False]

    km_t = KaplanMeierFitter()
    km_f = KaplanMeierFitter()

    km_t.fit(df_t['observed_mo'], df_t['died'])
    km_f.fit(df_f['observed_mo'], df_f['died'])

    lr = statistics.logrank_test(
        df_t['observed_mo'], df_f['observed_mo'],
        df_t['died'], df_f['died']
    )

    med_t, lo_t, hi_t = compute_median_ci(km_t)
    med_f, lo_f, hi_f = compute_median_ci(km_f)

    return {
        'true_kmf': km_t,
        'false_kmf': km_f,
        'median_true': med_t,
        'median_false': med_f,
        'ci_true_lo': lo_t,
        'ci_true_hi': hi_t,
        'ci_false_lo': lo_f,
        'ci_false_hi': hi_f,
        'logrank_p_value': lr.p_value
    }

# ✅ KM PLOT (FIXED ALIGNMENT)
def plot_KM_curve(df, feat, ax):
    stats = km_analysis_stats(df, feat)

    pos, neg = PAPER_LABELS.get(feat, (feat, f'Not {feat}'))

    stats['true_kmf'].plot_survival_function(ax=ax, label=pos, ci_show=False, linewidth=2.5)
    stats['false_kmf'].plot_survival_function(ax=ax, label=neg, ci_show=False, linewidth=2.5)

    ax.set_title(f'Survival Functions: {pos}', fontsize=18, fontweight='bold')
    ax.set_xlabel('Time from initial oncology visit (months)', fontsize=14)
    ax.set_ylabel('Proportion alive', fontsize=14)

    ax.legend(loc='upper right')

    # ✅ 🔥 CRITICAL FIX: FIXED RANGE
    max_time = 80
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, 1)

    # ✅ FORCE EXACT TICKS
    xticks = [0, 20, 40, 60, 80]
    ax.set_xticks(xticks)

    # ✅ 🔥 ADD RISK TABLE AFTER LIMITS ARE SET
    add_at_risk_counts(
        stats['true_kmf'],
        stats['false_kmf'],
        xticks=xticks,
        ax=ax
    )

    # ✅ 🔥 VERY IMPORTANT: DO NOT USE large left padding
    plt.subplots_adjust(left=0.18, bottom=0.32)

    # ✅ ANNOTATION
    text = (
        f"{pos}: {fmt(stats['median_true'])} mo "
        f"(95% CI {fmt(stats['ci_true_lo'])}-{fmt(stats['ci_true_hi'])})\n"
        f"{neg}: {fmt(stats['median_false'])} mo "
        f"(95% CI {fmt(stats['ci_false_lo'])}-{fmt(stats['ci_false_hi'])})"
    )

    ax.text(
        0.98, 0.01,
        text,
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=9,
        linespacing=1.2,
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor='black',
            alpha=0.85
        )
    )

# ✅ MAIN
def survival_analysis(results_dir=RESULTS_DIR):
    df = preprocess()

    paper_dir = os.path.join(results_dir, 'paper_figures')
    os.makedirs(paper_dir, exist_ok=True)

    for feat in SOC_ISO_FEATS + PSYCH_DIST_FEATS:
        print(f"Plotting {feat}")

        fig, ax = plt.subplots(figsize=(8, 8))

        plot_KM_curve(df, feat, ax)

        fig.savefig(
            os.path.join(paper_dir, f'{feat}.png'),
            dpi=300
        )

        plt.close(fig)

    print("✅ Done!")

if __name__ == '__main__':
    survival_analysis()


