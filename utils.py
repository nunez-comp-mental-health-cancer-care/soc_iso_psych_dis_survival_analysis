"""
Collection of helper functions used for the analysis
"""

import datetime
import numpy as np
from globals import DAYS_IN_MONTH

def parse_multi_to_datetime(str):
    """
    Parse the different datatime formats in our data and return a datetime object

    Warning: currently, no formats are similiar. But, if formats were similiar,
    could have an issue where d/m vs m/d ordering messes things up, e.g. parses
    as one option, then goes on to the next, parsing things as differently when
    one would count all. If occurs, will need to split up function.

    :param str: Datetime in one of the formats
    :return: datetime object
    """

    try:
        return datetime.datetime.strptime(str, '%m/%Y').date() # Month year
    except:
        try:
            return datetime.datetime.strptime(str, "%a %b %d %H:%M:%S %Y").date() # Month year day with hour second millisecond
        except:
            try:
                return datetime.datetime.strptime(str, "%Y-%m-%d").date() # Month year day
            except:
                try:
                    return datetime.datetime.strptime(str + '000', "%Y-%m-%d %H:%M:%S.%f").date()
                except:
                    try:
                        return datetime.datetime.strptime(str, "%m/%d/%Y %H:%M:%S").date()
                    except:
                        try:
                            return datetime.datetime.strptime(str, "%d-%b-%y").date()
                        except:
                            if np.isnan(str):
                                return str
                            else:
                                raise ValueError(f'Tried to parse a date string, but did not match known format. str: {str}')
                        #try:
                        #    return datetime.datetime.strptime(str, "%d/%m/%Y %H:%M:%S").date()
                        #except:
                        #    raise ValueError(f'Tried to parse a date string, but did not match known format. str: {str}')



#
# def bccstr_to_datetime(str):
#     """
#     :param str: In the provided BC Cancer format '%m/%Y'
#     :return: datatmine object using strptime
#     """
#     dt =
#
#     return dt
#
# def dtstr_to_datetime(str):
#     """
#     Needed because storing a datetime in pandas converts it to str
#     :param str: In default datatime format "%a %b %d %H:%M:%S %Y"
#     :return: datatmine object using strptime
#     """
#     dt = datetime.datetime.strptime(str, "%a %b %d %H:%M:%S %Y").date()
#
#     return dt
#
# def dcdtstr_to_datetime(s):
#     """
#     Converts the doc_details time format to strptime
#     :param str: In default datatime format "%Y-%m-%d %H:%M:%S.%f"
#     :return: datatmine object using strptime
#     """
#     dt = datetime.datetime.strptime(s + '000', "%Y-%m-%d %H:%M:%S.%f").date()
#
#     return dt

def dif_in_months(row, start_col='diag_mo_yr', end_col='death_mo_yr'):
    """
    :param row containing a start and end date to calculate month survival from, needs blanks filled by most recent date
    :return: difference in months
    """
    #start_date = datetime.datetime.strptime(row['diag_mo_yr'],'%m/%Y').date()
    #end_date = datetime.datetime.strptime(row['death_mo_yr'], '%m/%Y').date()
    start_date = parse_multi_to_datetime(row[start_col])
    end_date = parse_multi_to_datetime(row[end_col])

    delta = end_date - start_date

    #return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    return(round(delta.days/DAYS_IN_MONTH, 3)) # Return months surivived, up to 3 decimal places

def percentage_survivors(mo_survival, df):
    """
    :param mo_surv: the month cut off e.g. 12 = those who survived at least one year
    :param df: dataframe containing a column with month survival
    :return: percentage of patients who survived this long or greater
    """
    survivors = df.apply(lambda x: x['survival_mo'] >= mo_survival, axis=1)
    survived = len(survivors[survivors == True].index)

    all = len(df.index)

    return survived/all



def add_cols(cols, df):
    """
    Helper function that returns a col repersenting how many y's each subject had from the provided column names

    For adding up the PSSCAN subscores, number of Y's a pt gave

    :param cols: array of column names for this psscan subscore
    :param df: df with data
    :return: col (series) repersenting the sum of the number of cols == Y.
    """

    df['temp'] = 0
    for col in cols:
        numeric_col = df[col].replace('Y', 1).fillna(0)
        df['temp'] = df['temp'] + numeric_col

    return df['temp']