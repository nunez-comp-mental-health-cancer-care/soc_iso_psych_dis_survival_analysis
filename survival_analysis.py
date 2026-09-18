from preprocesser import preprocess
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib
matplotlib.use('Agg')
# --- Vector output settings (journal requires editable, image-free figures) ---
# fonttype 42 embeds TrueType so text remains selectable/editable text in the
# PDF rather than being converted to outlines or pixels.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import pandas as pd
import os

from globals import SOC_ISO_FEATS, PSYCH_DIST_FEATS, RESULTS_DIR

# =============================
# JCO COLORS (FINAL)
# =============================
ORANGE = "#E69F00"
BLUE = "#0072B2"

# =============================
# PANEL LABELS (FINAL)
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

# Panel order within each published figure.
FIGURE1_FEATS = ['LiveAlone', 'LostSpousePartner', 'NoRegularContact',
                 'NoEmotionalSupport', 'NoAssistance']
FIGURE2_FEATS = ['DepressionScore_CAT2', 'AnxietyScore_CAT2']

# =============================
# GROUP LABELS
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
# HELPERS
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
# MEDIAN + CI
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
# KM STATS
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
# KM PLOT
# =============================
def plot_KM_curve(df, feat, ax, annotation_fontsize=9, panel_fontsize=14):
    """Draw one KM panel into `ax`, with its at-risk table directly beneath.

    Returns the at-risk table axis so callers can adjust it if needed.
    """
    stats = km_analysis_stats(df, feat)

    neg_label, pos_label = PAPER_LABELS.get(feat, (f"Not {feat}", feat))

    fig = ax.figure
    n_axes_before = len(fig.axes)

    # BLUE FIRST (TOP OF LEGEND)
    stats["true_kmf"].plot(
        ax=ax, color=BLUE, linewidth=2.5, label=pos_label, ci_show=False
    )

    # ORANGE SECOND (BOTTOM)
    stats["false_kmf"].plot(
        ax=ax, color=ORANGE, linewidth=2.5, label=neg_label, ci_show=False
    )

    # AXES
    ax.set_xlabel("Time from initial oncology visit (months)")
    ax.set_ylabel("Proportion alive")

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .2, .4, .6, .8, 1])

    ax.legend(loc='upper right')

    # PANEL LABEL (OUTSIDE BOX)
    if feat in PANEL_MAP:
        ax.text(
            0.0, 1.04,
            PANEL_MAP[feat],
            transform=ax.transAxes,
            fontsize=panel_fontsize,
            fontweight='bold',
            ha='left',
            va='bottom'
        )

    # RISK TABLE (BLUE FIRST)
    add_at_risk_counts(
        stats["true_kmf"],
        stats["false_kmf"],
        ax=ax,
        xticks=[0, 20, 40, 60, 80, 100],
        rows_to_show=["At risk", "Censored", "Events"],
        labels=[pos_label, neg_label],
        at_risk_count_from_start_of_period=True
    )

    # COLOR-MATCH RISK TABLE LABELS
    # Capture the axis add_at_risk_counts just created, rather than assuming it
    # is the figure's last axis. In a multi-panel figure the previous
    # `plt.gcf().axes[-1]` approach would recolour the wrong panel's table.
    new_axes = fig.axes[n_axes_before:]
    table_ax = new_axes[-1] if new_axes else fig.axes[-1]
    for t in table_ax.texts:
        if t.get_text() == pos_label:
            t.set_color(BLUE)
        elif t.get_text() == neg_label:
            t.set_color(ORANGE)

    # ANNOTATION
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
        fontsize=annotation_fontsize,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )

    return table_ax

# =============================
# SAVE HELPER
# =============================
def save_vector(fig, path_no_ext, also_png=True):
    """Write a true-vector PDF for submission (and optionally a PNG preview).

    Only the PDF should be sent to the journal. EPS is deliberately not written:
    the white annotation box uses alpha transparency, which EPS cannot represent
    and which matplotlib may flatten to a raster block on EPS export.
    """
    pdf_path = f"{path_no_ext}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    if also_png:
        fig.savefig(f"{path_no_ext}.png", dpi=300, bbox_inches='tight')
    return pdf_path

# =============================
# COMBINED MULTI-PANEL FIGURES
# =============================
def build_combined_figure(df, feats, ncols, out_path_no_ext,
                          panel_w=8.0, panel_h=8.0):
    """Render several KM panels into ONE figure saved as a single vector PDF.

    Building the multi-panel figure here (rather than pasting exported panels
    together in Word) is what keeps the submitted file free of imported images.
    """
    nrows = -(-len(feats) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows))
    axes = axes.ravel() if hasattr(axes, 'ravel') else [axes]

    for ax, feat in zip(axes, feats):
        plot_KM_curve(df, feat, ax)

    # Hide any unused grid cell (e.g. the 6th slot when there are 5 panels).
    for ax in axes[len(feats):]:
        ax.set_visible(False)

    fig.subplots_adjust(hspace=0.75, wspace=0.28)
    path = save_vector(fig, out_path_no_ext)
    plt.close(fig)
    return path

# =============================
# MAIN
# =============================
def survival_analysis(results_dir=RESULTS_DIR):
    df = preprocess()

    paper_dir = os.path.join(results_dir, "paper_figures")
    os.makedirs(paper_dir, exist_ok=True)

    # ---- Individual panels (vector PDF each) ----
    for feat in SOC_ISO_FEATS + PSYCH_DIST_FEATS:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_KM_curve(df, feat, ax)
        fig.subplots_adjust(top=0.92, left=0.18, bottom=0.35)
        save_vector(fig, os.path.join(paper_dir, feat))
        plt.close(fig)

    # ---- Combined figures, one vector PDF per published figure ----
    fig1 = build_combined_figure(df, FIGURE1_FEATS, ncols=2,
                                 out_path_no_ext=os.path.join(paper_dir, "Figure_1"))
    fig2 = build_combined_figure(df, FIGURE2_FEATS, ncols=2,
                                 out_path_no_ext=os.path.join(paper_dir, "Figure_2"))

    print("Vector figures written to:", paper_dir)
    print("  Submit:", fig1)
    print("  Submit:", fig2)
    print("  (Individual panel PDFs also written, if the journal prefers them.)")
    print("  PNG copies are previews only - do not submit.")

if __name__ == "__main__":
    survival_analysis()
