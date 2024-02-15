import pandas as pd

def adult_data_cleansing():
    target_column = 'income'
    data = pd.read_csv('../data/adult.csv')
    data = data.replace(r'\s', '', regex=True)
    data[target_column] = (data[target_column] == '>50K').astype(int)
    data = data[~data.apply(lambda row: ' ?' in row.values, axis=1)]
    data = data.drop(['fnlwgt', 'capital_loss', 'capital_gain', 'education'], axis=1)
    category_mapping = {
        'Private': 'private',
        'Self-emp-not-inc': 'self',
        'Self-emp-inc': 'self',
        'Federal-gov': 'gov',
        'Local-gov': 'gov',
        'State-gov': 'gov',
        'Without-pay': 'other',
        'Never-worked': 'other'
    }
    data['workclass'] = data['workclass'].replace(category_mapping)
    mask = data['marital_status'].str.contains('Married', case=True)
    data.loc[mask, 'marital_status'] = 'Married'
    data.to_csv('../data/adult_cleaned.csv', index=False)


def compas_data_cleansing():
    data = pd.read_csv('../data/compas.csv')
    data = data[(data["days_b_screening_arrest"] <= 30)
                & (data["days_b_screening_arrest"] >= -30)
                & (data["is_recid"] != -1)
                & (data["c_charge_degree"] != 'O')
                & (data["score_text"] != 'N/A')].reset_index(drop=True)
    data.to_csv('../data/compas_cleaned.csv', index=False)


def communities_data_cleaning():
    target_column = 'income'
    data = pd.read_csv('../data/communities.csv', header=None)
    data = data.replace(r'\s', '', regex=True)
    data = data.loc[:, ~data.apply(lambda x: x.astype(str).str.contains('\?')).any()]
    data = data.drop([3], axis=1)
    # data[target_column] = (data[target_column] == '>50K').astype(int)
    # data = data[~data.apply(lambda row: ' ?' in row.values, axis=1)]
    # data = data.drop(['fnlwgt', 'capital_loss', 'capital_gain', 'education'], axis=1)
    # category_mapping = {
    #     'Private': 'private',
    #     'Self-emp-not-inc': 'self',
    #     'Self-emp-inc': 'self',
    #     'Federal-gov': 'gov',
    #     'Local-gov': 'gov',
    #     'State-gov': 'gov',
    #     'Without-pay': 'other',
    #     'Never-worked': 'other'
    # }
    # data['workclass'] = data['workclass'].replace(category_mapping)
    # mask = data['marital_status'].str.contains('Married', case=True)
    # data.loc[mask, 'marital_status'] = 'Married'
    data.to_csv('../data/communities_cleaned.csv', index=False)

#adult_data_cleansing()
#compas_data_cleansing()
#communities_data_cleaning()
