# Social Isolation, Psychological Distress, and Survival in Cancer

Analysis code accompanying our published study of whether social isolation and psychological distress, reported by patients at cancer diagnosis, are associated with overall survival. It is shared so readers can review and replicate the analyses. It uses patient-reported screening data from PRISM/PSSCAN (Psychosocial Screen for Cancer) linked to cancer registry outcomes.

**What the code does:**

- **Kaplan–Meier analyses** of overall survival for each social isolation domain and for psychological distress.
- **A multivariable Cox proportional hazards model** with all predictors entered together, adjusted for age, sex and metastatic status. The primary model stratifies by tumour group. A sensitivity model enters tumour group as a covariate instead.
- **A composite social isolation score** (0, 1, 2, 3+), with dose-response likelihood-ratio tests.
- **Proportional hazards and collinearity checks** for each model.

## Data availability

**This repository contains no patient data.** The PRISM/PSSCAN dataset holds identifiable health information. It cannot be shared publicly and is available only through the appropriate data access and ethics approvals.

The code expects the raw extract at `data/ALLPRISM.csv`. Everything under `data/` and `results/` is git-ignored, so neither the input data nor any output generated from it gets committed.

## Variables

| Group | Variables |
|---|---|
| Social isolation domains | Lives alone; recent loss of spouse or partner; no regular social contact; no emotional support; no assistance with daily activities (IADL) |
| Psychological distress | Anxiety symptoms; depression symptoms |
| Confounders | Age at diagnosis, sex, metastatic status at diagnosis, tumour group |
| Outcome | Months from diagnosis to death or last follow-up; vital status |

The column names and coding expected in the raw file are defined in [`globals.py`](globals.py).

## Setup

You need Python 3.8 or newer.

```bash
pip install -r requirements.txt
```

Or recreate the original conda environment:

```bash
conda env create -f environment.yml
```

## Usage

Run every script from the repository root.

1. Preprocess the raw extract. This writes `data/preprocessed_pts.csv` and QC summaries:
   ```bash
   python preprocesser.py
   ```
2. Build the descriptive statistics and the patient characteristics table:
   ```bash
   python dataset_analysis.py
   python dataset_analysis_supplementary.py  # tumour-site breakdown and staging detail
   ```
3. Run the survival analyses:
   ```bash
   python survival_analysis.py                                  # KM curves per domain, paper figures
   python "survival_analysis_stratified_covariate sensitivity.py" # combined Cox model, primary + sensitivity
   python survival_analysis_composite.py                        # composite score, dose-response
   python survival_analysis_combined.py                         # combined Cox summary
   ```
4. Check the model assumptions:
   ```bash
   python assumptions_check.py
   ```

Outputs, including figures as PDF and PNG and tables as CSV, are written to `results/`.

## Repository structure

| File | Purpose |
|---|---|
| `globals.py` | Paths, feature lists, and value-coding maps |
| `preprocesser.py` | Cleans and recodes the raw extract; writes QC summaries |
| `utils.py` | Date parsing and survival-time helpers |
| `dataset_analysis.py` | Cohort description and patient characteristics table |
| `dataset_analysis_supplementary.py` | Supplementary cohort tables (tumour site, staging) |
| `survival_analysis.py` | Kaplan–Meier curves and publication figures |
| `survival_analysis_stratified_covariate sensitivity.py` | Multivariable Cox model, stratified (primary) and covariate (sensitivity) |
| `survival_analysis_composite.py` | Composite social isolation score analyses |
| `survival_analysis_combined.py` | Combined Cox model summary |
| `assumptions_check.py` | Proportional hazards, collinearity, and imputation diagnostics |

## License

Released under the [MIT License](LICENSE).
