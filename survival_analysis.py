from preprocesser import preprocess
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
import pandas as pd
import os

from globals import SOC_ISO_FEATS, PSYCH_DIST_FEATS, RESULTS_DIR

# =============================
# ✅ JCO COLORS (FINAL)
# =============================
ORANGE = "#E69F00"
BLUE = "#0072B2"

# =============================
# ✅ PANEL LABELS (FINAL)
# =============================
PANEL_MAP = {
    'LiveAlone': '(A) Living alone',
    'LostSpousePartner': '(B) Recent partner loss',
    'NoRegularContact': '(C) No regular social contact',
    'NoEmotionalSupport': '(D) No emotional support',
    'NoAssistance': '(E) No IADL assistance',

    # Figure 2
    'DepressionScore_CAT2': '(A) Depression',
    'AnxietyScore_CAT2': '(B) Anxiety',
}

# =============================
# ✅ GROUP LABELS
# =============================
PAPER_LABELS = {
    'LiveAlone': ('Not living alone', 'Living alone'),
    'LostSpousePartner': ('No recent partner loss', 'Recent partner loss'),
    'NoRegularContact': ('Regular social contact', 'No regular social contact'),
    'NoEmotionalSupport': ('Has emotional support', 'No emotional support'),
    'NoAssistance': ('Has assistance with IADLs', 'No assistance with IADLs'),
    'Social_isolation_YN': ('No social isolation', 'Social isolation'),
    'AnxietyScore_CAT2': ('No anxiety', 'Anxiety'),
    'DepressionScore_CAT2': ('No depression', 'Depression'),
}

# =============================
# ✅ HELPERS
# =============================
def fmt(val):
    if val is None or pd.isna(val):
        return "NR"
    return f"{val:.1f}"

def order_ci(lo, hi):
    if lo is None or hi is None:
        return lo, hi
    return min(lo, hi), max(lo, hi)

# =============================
# ✅ MEDIAN + CI
# =============================
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
    lo = first_cross(upper_ci)
    hi = first_cross(lower_ci)

    lo, hi = order_ci(lo, hi)
    return median, lo, hi

# =============================
# ✅ KM STATS
# =============================
def km_analysis_stats(df, feat):
    temp = df[['observed_mo', 'died', feat]].copy().dropna()

    df_true = temp[temp[feat] == True]
    df_false = temp[temp[feat] == False]

    km_true = KaplanMeierFitter()
    km_false = KaplanMeierFitter()

    km_true.fit(df_true['observed_mo'], df_true['died'])
    km_false.fit(df_false['observed_mo'], df_false['died'])

    med_t, lo_t, hi_t = compute_median_ci(km_true)
    med_f, lo_f, hi_f = compute_median_ci(km_false)

    return {
        "true_kmf": km_true,
        "false_kmf": km_false,
        "median_true": med_t,
        "median_false": med_f,
        "ci_true_lo": lo_t,
        "ci_true_hi": hi_t,
        "ci_false_lo": lo_f,
        "ci_false_hi": hi_f
    }

# =============================
# ✅ KM PLOT
# =============================
def plot_KM_curve(df, feat, ax):
    stats = km_analysis_stats(df, feat)

    neg_label, pos_label = PAPER_LABELS.get(feat, (f"Not {feat}", feat))

    # ✅ BLUE FIRST (TOP OF LEGEND)
    stats["true_kmf"].plot(
        ax=ax, color=BLUE, linewidth=2.5, label=pos_label, ci_show=False
    )

    # ✅ ORANGE SECOND (BOTTOM)
    stats["false_kmf"].plot(
        ax=ax, color=ORANGE, linewidth=2.5, label=neg_label, ci_show=False
    )

    # ✅ AXES
    ax.set_xlabel("Time from initial oncology visit (months)")
    ax.set_ylabel("Proportion alive")

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .2, .4, .6, .8, 1])

    ax.legend(loc='upper right')

    # ✅ PANEL LABEL (OUTSIDE BOX)
    if feat in PANEL_MAP:
        ax.text(
            0.0, 1.04,
            PANEL_MAP[feat],
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            ha='left',
            va='bottom'
        )

    # ✅ RISK TABLE (BLUE FIRST)
    add_at_risk_counts(
        stats["true_kmf"],
        stats["false_kmf"],
        ax=ax,
        xticks=[0, 20, 40, 60, 80, 100],
        rows_to_show=["At risk", "Censored", "Events"],
        labels=[pos_label, neg_label],
        at_risk_count_from_start_of_period=True
    )

    # ✅ LAYOUT
    plt.subplots_adjust(top=0.92, left=0.18, bottom=0.35)

    # ✅ COLOR-MATCH RISK TABLE LABELS
    table_ax = plt.gcf().axes[-1]
    for t in table_ax.texts:
        if t.get_text() == pos_label:
            t.set_color(BLUE)
        elif t.get_text() == neg_label:
            t.set_color(ORANGE)

    # ✅ ANNOTATION
    txt = (
        f"{pos_label}: {fmt(stats['median_true'])} mo "
        f"(95% CI {fmt(stats['ci_true_lo'])}-{fmt(stats['ci_true_hi'])})\n"
        f"{neg_label}: {fmt(stats['median_false'])} mo "
        f"(95% CI {fmt(stats['ci_false_lo'])}-{fmt(stats['ci_false_hi'])})"
    )

    ax.text(
        0.98, 0.01,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )

# =============================
# ✅ MAIN
# =============================
def survival_analysis(results_dir=RESULTS_DIR):
    df = preprocess()

    paper_dir = os.path.join(results_dir, "paper_figures")
    os.makedirs(paper_dir, exist_ok=True)

    for feat in SOC_ISO_FEATS + PSYCH_DIST_FEATS:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_KM_curve(df, feat, ax)

        fig.savefig(
            os.path.join(paper_dir, f"{feat}.png"),
            dpi=300,
            bbox_inches='tight'
        )

        plt.close(fig)

    print("✅ FINAL JCO Oncology Practice–ready figures complete")

if __name__ == "__main__":
    survival_analysis()

    