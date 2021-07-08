
# Directories
RAW_PRISM_CSV = r'C:\Users\jjnunez\PRISM_data\ALLPRISM.csv' #csv containing the data for our analysis
DATA_DIR = r'C:\Users\jjnunez\PycharmProjects\PSSCAN_Soc_Iso\data\\'  # directory for intermediary data, raw data is stored elsewhere
RESULTS_DIR = r'C:\Users\jjnunez\PycharmProjects\PSSCAN_Soc_Iso\results\\'

# Columns
INDEX = "Other_Study_ID"

# Features to extract from the raw PRSIM file
SOC_ISO_FEATS_RAW = ['LiveAlone', 'LostSpousePartner', 'Assistance', 'RegularContact', 'EmotionalSupport', 'Social_isolation_YN']
PSYCH_DIST_FEATS = ['AnxietyScore_CAT2', 'DepressionScore_CAT2']
FROM_PRISM = ['PatStatus', 'Dx_to_death_lastfu_months', 'AgeAtDiagnosis', 'Sex', 'M1_IV_ALL_CAT'] + SOC_ISO_FEATS_RAW + PSYCH_DIST_FEATS

# Features to invert so all are True when having expected to be bad
SOC_ISO_FEATS_TO_INVERT = ['Assistance', 'RegularContact', 'EmotionalSupport']
SOC_ISO_FEATS_NO_INVERT = ['LiveAlone', 'LostSpousePartner', 'Social_isolation_YN']
SOC_ISO_FEATS = SOC_ISO_FEATS_NO_INVERT + ['NoAssitance', 'NoRegularContact', 'NoEmotionalSupport']

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
    'AnxietyScore_CAT2' : 'Anxiety category 2 or worse',
    'DepressionScore_CAT2' : 'Depression category 2 or worse'
    }