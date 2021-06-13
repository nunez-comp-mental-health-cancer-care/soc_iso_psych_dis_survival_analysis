import pandas as pd
import numpy as np
import warnings
#from data_globals import FROM_CANCER_DETAILS, INDEX, LATEST_DATE, FROM_PSSCAN_DETAILS, DISCIPLINES_OF_INTEREST, DISCIPLINE
#from data_globals import PSSCAN_EMOTS, PSSCAN_INFOS, PSSCAN_PRACS, PSSCAN_PHYIS, PSSCAN_SPIRS, PSSCAN_SOCLS, PSSCAN_ALL
#from data_globals import FROM_DOC_DETAILS, DATE_FEATURES
#from data_globals import RAW_CANCER_CSV, RAW_PSSCAN_CSV, RAW_DOC_CSV, DATA_DIR
from utils import dif_in_months, add_cols, parse_multi_to_datetime
from dateutil.relativedelta import relativedelta
from os import path
from datetime import timedelta
from globals import RAW_PRISM_CSV, DATA_DIR, INDEX, FROM_PRISM, SOC_ISO_FEATS

def preprocess(pts_csv=RAW_PRISM_CSV, data_dir=DATA_DIR, overwrite=False):


    filename = f'{data_dir}preprocessed_pts.csv'

    # Check if base preprocessed pts file already generated, return if so unless overwrite=True
    if path.exists(filename) and not(overwrite):
        print("preprocessed_pts.csv already generated, will load and exit preprocesser")
        return pd.read_csv(filename, index_col=INDEX)

    # Otherwise, generate
    print("preprocessed_pts.csv is processing, this may take a few minutes")
    pts_df = pd.read_csv(pts_csv, index_col=INDEX, engine='python')
    df = pts_df.copy()[FROM_PRISM] # Pull in only needed columns

    # Months observed use "Dx_to_death_lastfu_months" instead of calculating ourselves
    #df['obs_mo'] = df.apply(lambda row: dif_in_months(row, 'DiagnosisDate', 'Death_or_lastfu_date'), axis=1)
    #df = df.drop(columns=['DiagnosisDate', 'Death_or_lastfu_date'])

    # Fill blank Alive or Dead, Convert to True and False, replace blanks with False
    print(df[1250:1260]['PatStatus'])
    #df['PatStatus'] = df['PatStatus'].replace(r'^\s*$', False, regex=True)
    df['PatStatus'] = df['PatStatus'].fillna(False)
    df['PatStatus'] = df['PatStatus'].replace({"blank!": False, 'A': False, 'D': True})
    df = df.rename(columns={"PatStatus": "died", "Dx_to_death_lastfu_months": "observed_mo"})

    # For the various social isolation features, fill blanks with mode, and convert to True/False
    for feature in SOC_ISO_FEATS:
        df[feature] = df[feature].astype('str') # Convert col to string
        mode = df[feature].mode().iat[0]  # Find mode of column
        # Replace Blanks, coded as B or 99, with mode of the column, and convert Y N, 1 0 to True False
        df[feature] = df[feature].replace({"B": mode, "99": mode})
        df[feature] = df[feature].replace({"Y": True, "N": False, "1":True, "0":False, "B":mode, "99":mode})

    df.to_csv(filename)

    return df

def index_ends_10(row):
    """
    Quick helper function to make a random dspln column based on whether pt's index ends with 10.
    :param row: row from df
    :return: 1 if index ends with 10, 0 if not
    """
    if row.name % 10 == 0:
        return 1
    else:
        return 0

#
# def absolute_time_delta(x, pt_df):
#     """
#     Lambda function to return the absolute value of a timedelta between psscan and site admit dates
#     """
#     delta1 = parse_multi_to_datetime(x["screening_on"]) - parse_multi_to_datetime(pt_df.at[x.name, "site_admit_date"])
#     delta2 = parse_multi_to_datetime(pt_df.at[x.name, "site_admit_date"]) - parse_multi_to_datetime(x["screening_on"])
#
#     if delta1 > delta2:
#         return delta1
#     else:
#         return delta2
#
#
# def merge_psscan(pt_df, psscan_df):
#     """
#     Need a helper function to merge in psscan, because it has some duplicate values.
#     Merges psscan with pt_df, keep pts with either 1 psscan entiry, or if duplicates,
#     a psscan recorded within a month of the recorded first contact.
#
#
#     :param pt_df: patient dataframe
#     :param psscan_df: psscan dataframe
#     :return: merged dataframe
#     """
#
#     # Only those whose pts are in the pt_df
#     psscan_df = psscan_df[psscan_df.index.isin(pt_df.index)]
#
#     # First, start the no duplicate psscan by including all patients with no duplicate values
#     no_dup_psscan_df = psscan_df[~psscan_df.index.duplicated(keep=False)] # Remove pts with duplicate entires on PSSCAN
#
#
#     # Calculate the time difference for each psscan_row that was a duplicate
#     dup_psscan_df = psscan_df[psscan_df.index.duplicated(keep=False)].copy()
#     dup_psscan_df['psscan_admit_dif'] = dup_psscan_df.apply(lambda x: absolute_time_delta(x, pt_df), axis=1)
#
#
#     # Remote rows if more than a month from admit date
#     dup_psscan_df = dup_psscan_df[(dup_psscan_df['psscan_admit_dif'] <= timedelta(days=31))]
#     # Sort by time delta
#     dup_psscan_df = dup_psscan_df.sort_values('psscan_admit_dif')
#     # Drop duplicates from this, keep 1st which is closest to admit date
#     dup_psscan_df = dup_psscan_df[~dup_psscan_df.index.duplicated(keep='first')]
#
#     # Combine non-duplicate rows with these filters rows
#     final_psscan_df = pd.concat([no_dup_psscan_df, dup_psscan_df])
#     ##print(f'Concated psscan_dfs. Was {len(no_dup_psscan_df.index)} in no dups, then added filetered dups, now {len(final_psscan_df.index)}')
#
#     return pd.merge(pt_df, final_psscan_df, left_index=True, right_index=True, validate='1:1')
#
#
# def preprocess(pts_csv=RAW_PRISM_CSV, data_dir=DATA_DIR):
#
#
#     filename_base = f'{data_dir}preprocessed_pts.csv'
#
#     # Check if base preprocessed pts file already generated, return if so
#     if path.exists(filename_base):
#         print("preprocessed_pts.csv already generated, will load and exit preprocesser")
#         return pd.read_csv(filename_base, index_col=INDEX)
#
#     # Otherwise, generate
#     print("preprocessed_pts.csv is processing, this may take a few minutes")
#     pts_df = pd.read_csv(pts_csv, index_col=INDEX, low_memory=False)
#     df = pts_df.copy()[FROM_PRISM]
#
#
# def preprocess_pts_base(cancer_csv, psscan_csv, data_dir):
#     """
#     Creates a new preprocessed pts file, without applying any filters, and without needing
#     the preprocessed doc file to be made yet. Used that so this basefile can be used when
#     wanting to apply new filters, so that it isn't re-done everytime, and so that the preprocessed
#     docs can be made.
#
#     :param cancer_csv: path to file containting "cancer details", the main data for our pts
#     :param psscan_csv: path to file containing "psscan_details", the psscan data for our pts
#     :dev_mode: boolean, for development, only does first 100 pts so is quick
#     :return: dataframe of this preprocessed pt data, also writes it as csv to data_dir
#     """
#
#     filename_base = f'{data_dir}preprocessed_pts_base.csv'
#
#     # Check if base preprocessed pts file already generated, return if so
#     if path.exists(filename_base):
#         print("preprocessed_pts_base.csv already generated")
#         return pd.read_csv(filename_base, index_col=INDEX)
#
#     # Otherwise, generate
#     print("preprocessed_pts_base.csv is processing, this may take a few minutes")
#     cancer_df = pd.read_csv(cancer_csv, index_col='study_id', low_memory=False)
#     df = cancer_df.copy()[FROM_CANCER_DETAILS]
#
#     # Remove duplicate
#     # If patients have two cancers, they are in two values. Will need to think better how to handle this
#     # 62865 (entries) --> n=57,016
#     pt_count = len(df.index)
#     # Per Cheryl: We censored those folks. Methodologically you could either say only one cancer diagnosis within 5 years (excluding skin cancer).
#     # If the first cancer was greater than 5 years ago then you would select the second cancer as your starting point for analysis.
#     ##df = df.reset_index().drop_duplicates(subset=INDEX, keep=False).set_index(INDEX) # Need to do in this gross way as otherwise drop_duplicabtes ignores index
#     df = df[~df.index.duplicated(keep=False)]
#
#     print(f'Started with {pt_count} pts from mt2a_cancer_details.csv')
#     pt_count_after_dup_drop = len(df.index)
#     print(f'Dropped {pt_count - pt_count_after_dup_drop} pts after removing those with duplicate entires in cancer details. {len(df.index)} pts remain.')
#     ##print(f'Check to mak sure index now unique  {df.index.is_unique}')
#
#     # Add in relevant PSSCAN columns
#     psscan_df = pd.read_csv(psscan_csv, index_col='study_id', low_memory=False)
#     psscan_df = psscan_df.copy()[FROM_PSSCAN_DETAILS + PSSCAN_ALL]
#
#     df = merge_psscan(df, psscan_df)
#
#     print(f'Dropped {pt_count_after_dup_drop - len(df.index)} pts after removing those without PSSCAN, and those with duplicate entries that were not within a month of BCC admit. {len(df.index)} pts remain.')
#
#     # Convert date of diag and death columns to datetime formatted.
#     for feature in DATE_FEATURES:
#         print(f'{feature} has been converted')
#         df[feature] = df.apply(lambda row: parse_multi_to_datetime(row[feature]), axis=1)
#
#     ##print(f'Check to mak sure index still unique  {df.index.is_unique}')
#
#
#     # Make binary columns for anx and dep history
#     # Cutoffs from:
#     # Linden, W., Andrea Vodermaier, A., McKenzie, R. et al. The Psychosocial Screen for Cancer (PSSCAN): Further validation and normative data. Health Qual Life Outcomes 7, 16 (2009).
#     df['hx_dep'] = df[
#                        'past_year_two_weeks_depressed'] > 2  # Define depression as > 2, corresponding to "moderately" , "quite a bit", "very much"
#     df['hx_anx_8'] = df['score_anxiety'] >= 8  # Anxiety 8 cutoff per paper below corresponds to subclinical anxiety
#     df['hx_anx_11'] = df['score_anxiety'] >= 11  # Anxiety 11 cutoff per paper below corresponds to anxiety disorders
#     df['hx_dep_or_anx'] = df['hx_anx_8'] | df['hx_dep']
#
#     # Cutoffs from:
#     # Linden, W., Andrea Vodermaier, A., McKenzie, R. et al. The Psychosocial Screen for Cancer (PSSCAN): Further validation and normative data. Health Qual Life Outcomes 7, 16 (2009).
#
#     # Add column that specifies whether a patient lives alone or has no help
#     # Replace blanks with above columns with mode of the column
#     live_alone_mode = df['live_alone'].mode().iat[0] # Replace Blanks, coded as B, with mode of the column
#     assistance_mode = df['assistance'].mode().iat[0] # Replace Blanks, coded as B, with mode of the column
#
#     df['assistance'] = df['assistance'].replace({'B': assistance_mode})
#     df['live_alone'] = df['live_alone'].replace({'B': live_alone_mode})
#     df['live_alone'] = df['live_alone'].replace({"Y": True, "N": False}) # Convert Y to True, N to False, and B to blank
#     df['assistance'] = df['assistance'].replace({"Y": True, "N": False}) # As above
#
#     df['alone_or_no_assist'] = df.apply(lambda x: x['live_alone'] or not(x['assistance']), axis=1)# int(x['Price'].replace(',', '')), axis=1)
#
#     # Make a new column for the number of months a patient survives
#     df['died'] = ~df['death_mo_yr'].isnull()
#     ## df['death_mo_yr'].fillna(LATEST_DATE, inplace=True)  # No longer fill death date, keep na if not known to have died.
#     df['survival_mo'] = 'to_do' #df.apply(lambda row: dif_in_months(row), axis=1)
#
#
#     ##df['diag_mo_yr'] = df.apply(lambda row: parse_multi_to_datetime(row['diag_mo_yr']), axis=1)
#     ##df['death_mo_yr'] = df.apply(lambda row: parse_multi_to_datetime(row['death_mo_yr']), axis=1)
#
#     # Make a randomly assigned column selecting 10% for checking
#     df['RANDOMTENTH'] = df.apply(lambda row: index_ends_10(row),
#                                  axis=1)  # For this random column, TRUE if index is dividisble by 10.
#
#     # Add in their inital psscan need scores
#     # Emotional PSSCAN Subscore
#     df['emots'] = add_cols(PSSCAN_EMOTS, df)
#
#     # Information PSSCAN Subscore
#     df['infos'] = add_cols(PSSCAN_INFOS, df)
#
#     # Physical PSSCAN Subscore
#     df['phyis'] = add_cols(PSSCAN_PHYIS, df)
#
#     # Practical PSSCAN Subscore
#     df['pracs'] = add_cols(PSSCAN_PRACS, df)
#
#     # Social PSSCAN Subscore
#     df['socls'] = add_cols(PSSCAN_SOCLS, df)
#
#     # Spiritual PSSCAN Subscore
#     df['spirs'] = add_cols(PSSCAN_SPIRS, df)
#
#     # Make columns for total documents, and documents per month, as a stand-in for appointments/resource usage
#     warnings.warn(
#         "As we have only mo/yr of death and diag, docs per month is calculcated at this granularity. " +
#         "e.g. for those who survive 6 months, count as 7 months when calculating this ratio as could see someone for the rest of thier death month"
#     )
#
#     assert df.index.is_unique, 'patient dataframe does not have unique entries by study_id/index'
#     print(f'Baseline patient preprocessing complete. {len(df.index)} pts remain.')
#     #filename_base = filename = f'{data_dir}preprocessed_pts.csv'
#     df.to_csv(filename_base)
#     return df
#
#
# def preprocess_pts(cancer_csv=RAW_CANCER_CSV, psscan_csv=RAW_PSSCAN_CSV, doc_csv=RAW_DOC_CSV, data_dir=DATA_DIR, psyhx_subset='all', help_subset='all', min_mo=0):
#     """
#     Preprocesses the pt data, collecting data from the different sources, selecting only a subset as input, and filters ie
#     only those pts who survive at least 6 months and don't have duplicate entries.
#
#     :param psyhx_subset: 'all' to include all pts who survive 6 months and don't have duplicate entries, 'hx_anx_8' or 'hx_anx_11' if PSSCAN hx of anxiety, 'hx_dep' if history of depression'
#     :param help_subset: 'all' to include all pts who survive 6 months and don't have duplicate entries, 'alone', for pts who answer they live alone, 'no_assist' if no to assistance, 'alone_or_no_assist' for either
#     :param min_mo: Minimum number of months we know the pt survived for if they did not die. Exists because some pts in our database
#     were not observed for 5 years in our dataset, so we don't know if they have 5 year survival.
#     :return: preprocessed df for all the pts
#     """
#
#     # Start with Warnings to remind about TODOs etc
#     warnings.warn(
#         "For now, dropping subjects with duplicate cancer detail entries (2 or more cancers) or more than one entry on PSSCAN, could put in a more advanced system")
#
#
#     # First process a base file of all patients
#     df = preprocess_pts_base(cancer_csv, psscan_csv, data_dir)
#     # Once above generated, add in columns that required the docs to be processed.
#     # Split up as doc processing needs the base file
#     df = preprocess_pts_base_w_doc(df, doc_csv, data_dir)
#
#     # Set out filename for processed pt file
#     file_suffix = f'_{psyhx_subset}_{help_subset}_{min_mo}.csv'
#     filename = f'{data_dir}preprocessed_pts{file_suffix}'
#
#     # Check if file already exists, if so return and exit function
#     if path.exists(filename):
#         df = pd.read_csv(filename, index_col=INDEX)
#         print(f"preprocessed_pts{file_suffix} already generated")
#         return df
#
#     # Otherwise, proceed with applying specified filters
#
#     # Allow subsetting based on the person's availability of help/whether they live alone
#     if help_subset == 'alone':
#         df = df[(df['live_alone'])]
#     elif help_subset == 'no_assist':
#         df = df[~(df['assistance'])]
#     elif help_subset == 'alone_or_no_assist':
#         df = df[(df['alone_or_no_assist'])]
#     else:
#         assert help_subset == 'all', f'error, help_subset not expected value, is {help_subset}'
#
#
#     # Select subset of patients according to psych hx depending on subset argument
#     if psyhx_subset == 'hx_dep':
#         df = df[(df['hx_dep'])]
#     elif psyhx_subset == 'hx_anx_8':
#         df = df[(df['hx_anx_8'])]
#     elif psyhx_subset == 'hx_anx_11':
#         df = df[(df['hx_anx_11'])]
#     elif psyhx_subset == 'anx_or_dep':
#         df = df[(df['hx_dep_or_anx'])]
#     else:
#         assert(psyhx_subset == 'all')
#
#     # Remove patients who had less than 6 months survival n=57,016 --> n=51,933
#     ##df = df[(df['survival_mo'] >= 6)] Removed this constraint on Mar 22, 2021
#
#     # We may want to only choose pts that we know either died, or survived a specific time e.g. 5 years
#     # Needed as our dataset does not record all pts for 5 years
#     # Apply this filter if specified, no need to keep column. Giving 0 will make it essentially do nothing.
#
#     ##df['sufficient_obs'] = df.apply(lambda x: x['died'] or (x['survival_mo'] >= min_mo), axis=1)
#     ##df = df[df['sufficient_obs']]
#     ##df = df.drop(columns=['sufficient_obs'])
#
#     # Write to csv and return df
#     df.to_csv(filename)
#     return df
#
#
# def preprocess_pts_base_w_doc(df, doc_csv, data_dir):
#     """
#     Finished preprocessing the base patient file. Adds the columns that requires the documents to be preprocessed
#     first, or else we'd get a loop.
#
#     :param df: preprocessed pt dataframe, but without columns that required preprocessed docs
#     :param doc_csv: filename for the raw document csv, for passing to preprocess_docs
#     :param data_dir: data directory where the processed files are stored
#
#     :return: df with added columns using doc_df, including doc counts and whether they've seen a discipline
#     """
#
#     filename_base_w_docs = f'{data_dir}preprocessed_pts_base_w_docs.csv'
#
#     # Check if base preprocessed pts file already generated, return if so
#     if path.exists(filename_base_w_docs):
#         print("preprocessed_pts_base_w_docs.csv already generated")
#         return pd.read_csv(filename_base_w_docs, index_col=INDEX)
#
#     # Otherwise, generate
#     print("preprocessed_pts_base_w_docs.csv is processing, this may take a few minutes")
#
#     # Grab the preprocessed document dataframe now that that's been generated
#     doc_df = preprocess_docs(doc_csv=doc_csv)
#
#     # Make columns for whether they've seen specified providers
#     dsplns = DISCIPLINES_OF_INTEREST.copy()
#     dsplns.remove('RANDOMTENTH')  # RANDOMTENTH added above so don't need for below for loop
#
#     for dspln in dsplns:
#         dspln_df = saw_discipline(doc_df, dspln)
#         df = df.join(dspln_df, on=INDEX)
#         df[dspln].fillna(0, inplace=True)
#
#     ##doc_df = doc_df.dropna(0, 'any')
#     ##doc_df['date_of_event'] = doc_df.apply(lambda row: parse_multi_to_datetime(row['date_of_event']), axis=1)
#     ##df['docs6to18mo'] = df.apply(lambda row: docs6to18mo(doc_df, row), axis=1)
#     ##df['docs6to18mo_survival'] = df.apply(lambda row: min(row['survival_mo'] - 5, 11) + 1, axis=1)
#     ##df['docspermonth'] = df['docs6to18mo'] / df['docs6to18mo_survival']
#
#     df.to_csv(filename_base_w_docs)
#     return df
#
# def preprocess_docs(doc_csv=RAW_DOC_CSV, data_dir=DATA_DIR, cancer_csv=RAW_CANCER_CSV, psscan_csv=RAW_PSSCAN_CSV):
#     """
#     Preprocesses the document_details file to generate a df with needed details for further analysis
#     :param doc_csv: filename to q18-0193_document_details file, including path
#     :param data_dir: path to data directory used to store processeed data
#     :return: processed doc df, also writes out to data_dir
#     """
#
#
#     # Set out filename for preprocessed doc file
#     filename = f'{data_dir}preprocessed_docs.csv'
#
#     # Check if file already exists, if so return and exit function
#     if path.exists(filename): # and not(dev_mode):
#         df = pd.read_csv(filename)
#         print("Preprocessed docs dataframe previously generated, reading in file and exiting preprocess_docs")
#         return df
#
#     print("Generating new preprocessed_docs df as not previously generated")
#
#     df = pd.read_csv(doc_csv, index_col='document_id')#, low_memory=False)
#
#     # Get a df of patient details
#     pt_df = preprocess_pts_base(cancer_csv, psscan_csv, data_dir)
#
#     # SLP documents are misclassified, sometimes have blank fesr_discipline, or have it as NUTRITION.
#     # Per scratch, the 'document_type' column seems to always be correct, so will use than to rename fesr_discipline to SPEECH THRPY
#     # when SLP. Do before dropping as will replace some values
#     df.loc[df['document_type'] == 'SLP', 'fasr_discipline'] = 'SPEECH THRPY'
#
#     print(f'Started with {len(df)} docs from q18-0193_document_details.csv')
#     # Drop rows with null values
#     before_null_drop = len(df)
#     df = df.replace("NULL", np.nan)
#     df = df.replace("nan", np.nan)
#     df = df.dropna(0, 'any')
#     print(f'Dropped {before_null_drop - len(df)} documents due to null entries, {len(df)} docs remain')
#
#     # Drop rows if subjects were dropped ie due to not surviving 6mo or duplicate entries
#     before_subj_drop = len(df)
#     df = df[df['study_id'].isin(pt_df.index)]
#     print(f'Dropped {before_subj_drop - len(df)} documents due to dropped subjs, {len(df)} docs remain ')
#
#     #Only keep rows of documents by providers we are interested in
#     before_dspln_drop = len(df)
#     df = df[df['fasr_discipline'].isin(DISCIPLINES_OF_INTEREST)]
#     print(f'Dropped {before_dspln_drop - len(df)} documents due to dropped dsplns, {len(df)} docs remain ')
#
#     #Only keep columns of documents we're interested in
#     df = df[FROM_DOC_DETAILS]
#
#     # Get a dictionary of study_id to diagnosis date
#     id_to_diag = pd.Series(pt_df.diag_mo_yr.values, index=pt_df.index).to_dict()
#     # Use dictionary to write the diagnosis date of the person who each document is attached to
#     df['pt_diag_mo_yr'] = df['study_id'].map(id_to_diag)
#     # Calculate how many months from a pt's diagnosis a document was generate at
#     df['pt_mo_since_dx'] = df.apply(lambda row: dif_in_months(row, 'pt_diag_mo_yr', 'date_of_event'), axis=1)
#
#
#     # Docs are classified as both fasr_discipline DIETICIAN and NUTRITION, but usually NUTRITION. So, change all DIETICIAN to NUTRITION
#     df['fasr_discipline'] = df['fasr_discipline'].replace('DIETICIAN',"NUTRITION")
#
#     # Replace NULL With np.nan for more machine readability
#     df = df.replace("NULL", np.nan)
#
#
#     # Write to csv and return df
#     print(f'Document processing complete, {len(df.index)} documents remain.')
#     df.to_csv(filename)
#     return df

if __name__ == '__main__':
    preprocess(overwrite=True)
    # preprocess_pts(psyhx_subset='anx_or_dep')
    # preprocess_pts(help_subset='alone_or_no_assist')
    # preprocessed_df = preprocess_pts(cancer_csv, psscan_csv, doc_csv, data_dir, dev_mode=True)
    # preprocessed_docs = preprocess_docs()
    print("Pre-processing complete!")
