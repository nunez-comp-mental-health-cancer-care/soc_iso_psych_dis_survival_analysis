from pathlib import Path

# Directories
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = str(ROOT_DIR / 'data') + '/'
RESULTS_DIR = str(ROOT_DIR / 'results') + '/'
RAW_PRISM_CSV = str(Path(DATA_DIR) / 'ALLPRISM.csv')  # csv containing the data for our analysis

# Ensure local output directories exist
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# Columns
INDEX = "Other_Study_ID"

# Features to extract from the raw PRSIM file
SOC_ISO_FEATS_RAW = ['LiveAlone', 'LostSpousePartner', 'Assistance', 'RegularContact', 'EmotionalSupport', 'Social_isolation_YN']
PSYCH_DIST_FEATS = ['AnxietyScore_CAT2', 'DepressionScore_CAT2']
FROM_PRISM = [ 'Dx_to_death_lastfu_months', 'AgeAtDiagnosis', 'PatStatus', 'Sex', 'M1_IV_ALL_CAT'] + SOC_ISO_FEATS_RAW + PSYCH_DIST_FEATS

# Features to invert so all are True when having expected to be bad
SOC_ISO_FEATS_TO_INVERT = ['Assistance', 'RegularContact', 'EmotionalSupport']
SOC_ISO_FEATS_NO_INVERT = ['LiveAlone', 'LostSpousePartner', 'Social_isolation_YN']
SOC_ISO_FEATS = SOC_ISO_FEATS_NO_INVERT + ['NoAssistance', 'NoRegularContact', 'NoEmotionalSupport']
# The 5 individual social isolation domains used in multivariable models.
# Excludes Social_isolation_YN (composite binary flag) to avoid collinearity.
SOC_ISO_DOMAIN_FEATS = ['LiveAlone', 'LostSpousePartner', 'NoRegularContact', 'NoEmotionalSupport', 'NoAssistance']

# If want to calculate seperately, used following, slightly different from what exists
# 'DiagnosisDate', 'Death_or_lastfu_date']

# Constant of days per month
DAYS_IN_MONTH = 30.436875

# Renaming map for dataset analysis
DATASET_ANALYSIS_COL_MAP = {
    'PatStatus' : 'Alive at end of fu',
    'Dx_to_death_lastfu_months' : 'Months until death or last follow-up',
    'Sex' : 'Male',
    'M1_IV_ALL_CAT' : 'MetastaticAtDiagnosis',
    'Social_isolation_YN' : 'Any social isolation',
    'AnxietyScore_CAT2' : 'Anxiety',
    'DepressionScore_CAT2' : 'Depression'
    }

# True value map for binary columns for dataset analysis
TRUE_COL_MAP = {
    'PatStatus' : 'A',
    'LiveAlone': 'Y',
    'LostSpousePartner': 'Y',
    'Assistance': 'Y',
    'RegularContact': 'Y',
    'EmotionalSupport': 'Y',
    'Social_isolation_YN': 1,
    'AnxietyScore_CAT2': 2.0,
    'DepressionScore_CAT2' : 2.0,
    'M1_IV_ALL_CAT' : 1,
    'Sex' : 'M'
}

BLANK_COL_MAP = {
    'PatStatus' : 'no blanks',
    'Sex' : 'no blanks',
    'LiveAlone': 'B',
    'LostSpousePartner': 'B',
    'Assistance': 'B',
    'RegularContact': 'B',
    'EmotionalSupport': 'B',
    'Social_isolation_YN': 99,
    'AnxietyScore_CAT2': 0.0,
    'DepressionScore_CAT2' : 0.0,
    'M1_IV_ALL_CAT' : 99,

}
