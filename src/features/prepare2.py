# TODO: ordenar os imports
# TODO: install xgboost, lightgbm, catboost
# TODO: conda remove seaborn, conda install seaborn=0.9.0
# TODO: pip install dask
# TODO: utils
import gc
import os.path
from multiprocessing import cpu_count
import pandas as pd
from dask import dataframe as dd
from dask.multiprocessing import get
from functools import reduce


def normalize_months(df,
                     id_col='ids',
                     month_col='month',
                     normalized_month_col='normalized_month'):
    """Set the max month of and ID in the dataset equals the max month of the dataset

    Find the max month of a datset and normalizes it backwards.
    Each ID of the dataset starts with month 0, but has a different number of months,
    but the last month of each ID represents the last month of the dataset,
    so we set this month as the max of the dataset, and recount other months backwards.

    We can find were is the very first month of each ID in the timeline.

    Example:
    a) The dataset has 36 months (0-35)
    b) the ID 1234 has 10 months in dataset (0-9)
    So the normalized months of ID 1234 will be 26-35. The ID 1234 become a client in month 26

    Parameters:
        - df: a pandas dataframe
        - id_col: the id column
        - month_col: the month column
        - normalized_month_col: the name of new column which will receive normalized months

    Retun:
        This function changes the dataset inplace and do not returns anything
    """
    # initiaizes new column with actual months
    df[normalized_month_col] = df[month_col]

    # max month in dataframe
    max_month = max(df[month_col])

    # set of unique ids in dataset
    ids_set = set(df[id_col])

    for current_id in ids_set:
        # subset of dataset with only the current id
        ids_subset = df[df[id_col] == current_id]

        # max month of current id
        max_current_id_month = max(ids_subset[month_col])

        # diff between max month of dataset and max month of current id
        diff_month = max_month - max_current_id_month

        # update months:
        for index, row in ids_subset.iterrows():
            # df.set_value(index, normalized_month_col, row[month_col] + diff_month)
            df.at[index, normalized_month_col] = row[month_col] + diff_month


def prep_spend(df):
    inflation = .005
    exchange_rate = .05
    interest_rate = .17

    id_col = 'ids'
    month_col = 'month'
    normalized_month_col = 'normalized_month'

    # initiaizes new column with actual months
    df[normalized_month_col] = df[month_col]
    df['revolving_months_in_a_row'] = 0
    df['revolving_min_months_in_a_row'] = 0
    df['invoice'] = 0
    df['income_spend'] = 0
    df['income_interest'] = 0
    df['interest'] = 0
    df['p_paid'] = 0
    df['p_spend'] = 0

    # max month in dataframe
    max_month = max(df[month_col])

    # set of unique ids in dataset
    ids_set = set(df[id_col])

    for current_id in ids_set:
        # subset of dataset with only the current id
        ids_subset = df[df[id_col] == current_id].sort_values(by='month')

        # max month of current id
        max_current_id_month = max(ids_subset[month_col])

        # diff between max month of dataset and max month of current id
        diff_month = max_month - max_current_id_month

        months_count = 0
        months_min_count = 0
        last_revolving_balance = 0
        # previous_revolved = False
        # update months:
        for index, row in ids_subset.iterrows():
            if row['revolving_balance'] <= 0:
                months_count = 0
                months_min_count = 0
            else:
                months_count += 1
                if row['revolving_balance'] > row['spends'] * 0.9:
                    months_min_count += 1
                else:
                    months_min_count = 0

            previous_debt = last_revolving_balance * (1 + interest_rate)

            # df.set_value(index, normalized_month_col, row[month_col] + diff_month)
            # df.set_value(index, 'revolving_months_in_a_row', months_count)
            # df.set_value(index, 'revolving_min_months_in_a_row', months_min_count)
            df.at[index, normalized_month_col] = row[month_col] + diff_month
            df.at[index, 'revolving_months_in_a_row'] = months_count
            df.at[index, 'revolving_min_months_in_a_row'] = months_min_count
            df.at[index, 'invoice'] = row['spends'] + previous_debt
            df.at[index, 'income_spend'] = row['spends'] * (exchange_rate) * (
                1 - inflation)
            df.at[
                index,
                'income_interest'] = last_revolving_balance * interest_rate * (
                    1 - inflation)
            df.at[index, 'interest'] = last_revolving_balance * interest_rate
            # df.at[index, 'p_paid'] = 0
            # df.at[index, 'p_spend'] = 0

            last_revolving_balance = row['revolving_balance']


def classify_by_income(row):
    """Classify applicants by income

    This is not the real classification used in Brazil,
    but it can get closer to our current scenario,
    and it's good enough to our analysis

    Params:
        - row: a pandas dataframe row

    Returns:
        - A modified pandas dataframe row
    """

    min_salary = 954.0
    monthly_income = row['income'] / 12

    if monthly_income <= min_salary * 2:
        income_class = 'E'
    elif monthly_income <= min_salary * 4:
        income_class = 'D'
    elif monthly_income <= min_salary * 10:
        income_class = 'C'
    elif monthly_income <= min_salary * 20:
        income_class = 'B'
    else:
        income_class = 'A'

    row['class'] = income_class
    return row


def calculate_member_since(row, spend):
    """Calculate when a customer become member

    Params:
        - row: a pandas dataframe row
        - spend: a pandas dataframe which has spend information of each customer by month

    Returns:
        - A modified pandas dataframe row
    """

    try:
        current_id_min_month = min(
            spend[spend['ids'] == row['ids']]['normalized_month'])
    except:
        print('Except', row['ids'])
        current_id_min_month = 0

    row['member_since'] = current_id_min_month
    return row


def acquisition_init_calculated_columns(df):
    df['class'] = ''
    df['member_since'] = 0
    df['total_spent'] = 0
    df['total_revolving'] = 0
    df['total_minutes'] = 0
    df['total_card_requests'] = 0
    df['total_months'] = 0
    df['total_revolving_months'] = 0
    df['total_months_spent_too_much'] = 0
    df['total_revolving_min_months'] = 0
    df['max_revolving_months_in_a_row'] = 0
    df['max_revolving_min_months_in_a_row'] = 0
    return df


def try_min(df, new_col, row, base_col):
    try:
        val = min(df[df['ids'] == row['ids']][base_col])
    except:
        print('Min Except', row['ids'])
        val = 0

    row[new_col] = val
    return row


def try_reduce(df, new_col, row, base_col):
    try:
        val = reduce(lambda x, y: x + y, df[df['ids'] == row['ids']][base_col])
    except:
        print('Reduce Except', row['ids'])
        val = 0

    row[new_col] = val
    return row


def acquisition_calculations(row, spend):
    """Create calculated columns on acquisition dataframe

    Parameters:
        - row: pandas dataframe row

    Return:
        A modified oandas dataframe row with calculated columns
    """

    try:
        credit_line = spend[spend['ids'] == row['ids']].sort_values(
            by='month')['credit_line'][0]
    except:
        credit_line = -1

    row['credit_line'] = credit_line
    return (row)


if __name__ == '__main__':
    N_CORES = cpu_count()
    BASE_PATH = NEW_DATA_PATH = '../../data/interim/'

    spend = pd.read_csv(NEW_DATA_PATH + 'spend_normalized_month.csv')

    # Create calculated columns for Acquisitions if it was not created before
    if os.path.isfile(NEW_DATA_PATH + 'acquisition_train_calculated_v2.csv'):
        print('acquisition_train_calculated.csv Found!')
        acquisition = pd.read_csv(NEW_DATA_PATH +
                                  'acquisition_train_calculated_v2.csv')
    else:
        print('Creating acquisition_train_calculated_v2.csv')
        acquisition = pd.read_csv(BASE_PATH +
                                  'acquisition_train_calculated.csv')
        acquisition = acquisition_init_calculated_columns(acquisition)
        # acquisition = acquisition.apply(acquisition_calculations, args=(spend,), axis=1)
        acquisition = dd.from_pandas(acquisition, npartitions=N_CORES)\
                        .map_partitions(lambda df, spend=spend: df.apply(acquisition_calculations, args=(spend,), axis=1))\
                        .compute(get=get)
        acquisition.to_csv(NEW_DATA_PATH +
                           'acquisition_train_calculated_v2.csv')
        print('acquisition_train_calculated_v2.csv Created')
