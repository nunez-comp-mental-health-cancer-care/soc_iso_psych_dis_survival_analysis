
# Directories
RAW_PRISM_CSV = r'C:\Users\jjnunez\PRISM_data\ALLPRISM.csv' #csv containing the data for our analysis
DATA_DIR = r'C:\Users\jjnunez\PycharmProjects\PSSCAN_Soc_Iso\data\\'  # directory for intermediary data, raw data is stored elsewhere
RESULTS_DIR = r'C:\Users\jjnunez\PycharmProjects\PSSCAN_Soc_Iso\results\\'

# Columns
INDEX = "Other_Study_ID"

# Features to extract from the raw PRSIM file
SOC_ISO_FEATS = ['LiveAlone', 'LostSpousePartner', 'Assistance', 'RegularContact', 'EmotionalSupport', 'Social_isolation_YN']
FROM_PRISM = ['PatStatus', 'Dx_to_death_lastfu_months', 'AgeAtDiagnosis', 'Sex', 'M1_IV_ALL_CAT'] + SOC_ISO_FEATS

# If want to calculate seperately, used following, slightly different from what exists
# 'DiagnosisDate', 'Death_or_lastfu_date']

# Constant of days per month
DAYS_IN_MONTH = 30.436875