# -*- coding: utf-8 -*-
"""Pre-processing the dataset."""
import gc
import os.path
from functools import reduce
from multiprocessing import cpu_count

import pandas as pd
from dask import dataframe as dd
from dask.multiprocessing import get


def prep_spend(df):
    """Pre-processing spend dataset."""
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
            invoice = row['spends'] + previous_debt
            paid = invoice - row['revolving_balance']

            df.at[index, normalized_month_col] = row[month_col] + diff_month
            df.at[index, 'revolving_months_in_a_row'] = months_count
            df.at[index, 'revolving_min_months_in_a_row'] = months_min_count
            df.at[index, 'invoice'] = invoice
            df.at[index, 'income_spend'] = row['spends'] * (exchange_rate) * (
                1 - inflation)
            df.at[
                index,
                'income_interest'] = last_revolving_balance * interest_rate * (
                    1 - inflation)
            df.at[index, 'interest'] = last_revolving_balance * interest_rate
            df.at[index, 'p_paid'] = paid / invoice * 100 if invoice > 0 else 0
            df.at[index, 'p_spend'] = row['spends'] / row['credit_line'] * 100

            last_revolving_balance = row['revolving_balance']


def classify_by_income(row):
    """Classify applicants by income.

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
    """Calculate when a customer become member.

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
    """Initialize new columns on acquisition dataset."""
    df['class'] = ''
    df['member_since'] = 0
    df['total_spent'] = 0
    df['total_revolving'] = 0
    df['total_minutes'] = 0
    df['total_card_requests'] = 0
    df['total_months'] = 0
    df['avg_spend'] = 0
    df['total_revolving_months'] = 0
    df['credit_line'] = 0
    return df


def try_min(df, new_col, row, base_col):
    """Try calc the min value of a column."""
    try:
        val = min(df[df['ids'] == row['ids']][base_col])
    except Exception as e:
        print('Min Except', row['ids'], e)
        val = 0

    row[new_col] = val
    return row


def try_reduce(df, new_col, row, base_col):
    """Try reduce the sum of a column."""
    try:
        val = reduce(lambda x, y: x + y, df[df['ids'] == row['ids']][base_col])
    except Exception as e:
        print('Reduce Except', row['ids'], e)
        if row['ids'] == 'foo':
            print(row)
        val = 0

    row[new_col] = val
    return row


def acquisition_calculations(row, spend):
    """Create calculated columns on acquisition dataframe.

    Parameters:
        - row: pandas dataframe row

    Return:
        A modified pandas dataframe row with calculated columns

    """
    id_slice = spend[spend['ids'] == row['ids']]
    if len(id_slice) == 0 and row['ids'] != 'foo':
        print(row['ids'])
        exit()

    row = classify_by_income(row)
    row = try_reduce(id_slice, 'total_spent', row, 'spends')
    row = try_reduce(id_slice, 'total_revolving', row, 'revolving_balance')
    row = try_reduce(id_slice, 'total_minutes', row, 'minutes_cs')
    row = try_reduce(id_slice, 'total_card_requests', row, 'card_request')
    row['total_months'] = len(id_slice)
    row['total_revolving_months'] = len(
        spend[(spend['ids'] == row['ids']) & (spend['revolving_balance'] > 0)])
    row = try_min(id_slice, 'member_since', row, 'normalized_month')
    
    if row['total_months'] > 0:
        row['avg_spend'] = row['total_spent'] / row['total_months']
    
    try:
        row['credit_line'] = id_slice.sort_values(
            by='month').reset_index()['credit_line'][0]
    except:
        row['credit_line'] = -1

    return (row)


if __name__ == '__main__':
    N_CORES = cpu_count()
    RAW_DATA_PATH = '../../data/raw/'
    INTERIM_DATA_PATH = '../../data/interim/'

    SPEND_FILE = 'spend_train.csv'
    ACQUISITION_FILE = 'acquisition_train.csv'

    spend = None
    # Create calculated columns for Spend dataset if it was not created before
    if os.path.isfile(os.path.join(INTERIM_DATA_PATH, SPEND_FILE)):
        print('{} Found!'.format(os.path.join(INTERIM_DATA_PATH, SPEND_FILE)))
        spend = pd.read_csv(os.path.join(INTERIM_DATA_PATH, SPEND_FILE))
    else:
        print('Creating {}'.format(
            os.path.join(INTERIM_DATA_PATH, SPEND_FILE)))
        spend = pd.read_csv(os.path.join(RAW_DATA_PATH, SPEND_FILE))
        
        prep_spend(spend)
        spend.to_csv(os.path.join(INTERIM_DATA_PATH, SPEND_FILE), index=False)

        print('{} Created'.format(os.path.join(INTERIM_DATA_PATH, SPEND_FILE)))

    # Create calculated columns for Acquisitions if it was not created before
    acquisition = None
    if os.path.isfile(os.path.join(INTERIM_DATA_PATH, ACQUISITION_FILE)):
        print('{} Found!'.format(
            os.path.join(INTERIM_DATA_PATH, ACQUISITION_FILE)))
        acquisition = pd.read_csv(
            os.path.join(INTERIM_DATA_PATH, ACQUISITION_FILE))
    else:
        print('Creating {}'.format(
            os.path.join(INTERIM_DATA_PATH, ACQUISITION_FILE)))
        acquisition = pd.read_csv(
            os.path.join(RAW_DATA_PATH, ACQUISITION_FILE))
        acquisition = acquisition_init_calculated_columns(acquisition)
        acquisition = dd.from_pandas(acquisition, npartitions=N_CORES)\
                        .map_partitions(lambda df, spend=spend:
                                        df.apply(acquisition_calculations,
                                                 args=(spend,), axis=1))\
                        .compute(get=get)
        acquisition.to_csv(
            os.path.join(INTERIM_DATA_PATH, ACQUISITION_FILE), index=False)
        print('{} Created'.format(
            os.path.join(INTERIM_DATA_PATH, ACQUISITION_FILE)))
