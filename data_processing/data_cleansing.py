import pandas as pd
from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSMobility, ACSTravelTime, ACSHealthInsurance
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



def folks_income_data_cleaning():
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    ca_data = data_source.get_data(states=["FL"], download=True)
    ACSIncome_categories = {
        "COW": {
            1.0: (
                "Employee of a private for-profit company or"
                "business, or of an individual, for wages,"
                "salary, or commissions"
            ),
            2.0: (
                "Employee of a private not-for-profit, tax-exempt,"
                "or charitable organization"
            ),
            3.0: "Local government employee (city, county, etc.)",
            4.0: "State government employee",
            5.0: "Federal government employee",
            6.0: (
                "Self-employed in own not incorporated business,"
                "professional practice, or farm"
            ),
            7.0: (
                "Self-employed in own incorporated business,"
                "professional practice or farm"
            ),
            8.0: "Working without pay in family business or farm",
            9.0: "Unemployed and last worked 5 years ago or earlier or never worked",
        },
        "SCHL": {
            1.0: "No schooling completed",
            2.0: "Nursery school, preschool",
            3.0: "Kindergarten",
            4.0: "Grade 1",
            5.0: "Grade 2",
            6.0: "Grade 3",
            7.0: "Grade 4",
            8.0: "Grade 5",
            9.0: "Grade 6",
            10.0: "Grade 7",
            11.0: "Grade 8",
            12.0: "Grade 9",
            13.0: "Grade 10",
            14.0: "Grade 11",
            15.0: "12th grade - no diploma",
            16.0: "Regular high school diploma",
            17.0: "GED or alternative credential",
            18.0: "Some college, but less than 1 year",
            19.0: "1 or more years of college credit, no degree",
            20.0: "Associate's degree",
            21.0: "Bachelor's degree",
            22.0: "Master's degree",
            23.0: "Professional degree beyond a bachelor's degree",
            24.0: "Doctorate degree",
        },
        "MAR": {
            1.0: "Married",
            2.0: "Widowed",
            3.0: "Divorced",
            4.0: "Separated",
            5.0: "Never married or under 15 years old",
        },
        "SEX": {1.0: "Male", 2.0: "Female"},
        "RAC1P": {
            1.0: "White alone",
            2.0: "Black or African American alone",
            3.0: "American Indian alone",
            4.0: "Alaska Native alone",
            5.0: (
                "American Indian and Alaska Native tribes specified;"
                "or American Indian or Alaska Native,"
                "not specified and no other"
            ),
            6.0: "Asian alone",
            7.0: "Native Hawaiian and Other Pacific Islander alone",
            8.0: "Some Other Race alone",
            9.0: "Two or More Races",
        },
    }
    X, y, _ = ACSIncome.df_to_pandas(ca_data, categories=ACSIncome_categories, dummies=True)
    for column in X.columns[5:]:
        X[column] = X[column].astype(int)
    X["income"] = y.astype(int)
    X.to_csv('../data/folks_income_FL_cleaned.csv', index=False)


def folks_travel_data_cleaning():
    # Fetching the data source as before
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    ca_data = data_source.get_data(states=["FL"], download=True)

    # Getting the features and labels for the Travel Time to Work task
    X_travel, y_travel, _ = ACSTravelTime.df_to_pandas(ca_data, dummies=True)
    X_travel["travel_time"] = y_travel
    for column in X_travel.columns:
        X_travel[column] = X_travel[column].astype(int)
    X_travel.to_csv('../data/folks_travel_FL_cleaned.csv', index=False)



def folks_mobility_data_cleaning():
    # Fetching the data source as before
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    ca_data = data_source.get_data(states=["FL"], download=True)

    X_travel, y_travel, _ = ACSMobility.df_to_pandas(ca_data, dummies=True)
    X_travel["mobility"] = y_travel
    for column in X_travel.columns:
        X_travel[column] = X_travel[column].astype(int)
    X_travel.to_csv('../data/folks_mobility_FL_cleaned.csv', index=False)


def folks_insurance_data_cleaning():
    # Fetching the data source as before
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    ca_data = data_source.get_data(states=["FL"], download=True)

    X_travel, y_travel, _ = ACSHealthInsurance.df_to_pandas(ca_data, dummies=True)
    X_travel["insured"] = y_travel
    for column in X_travel.columns:
        X_travel[column] = X_travel[column].astype(int)
    X_travel = X_travel.sample(frac = 0.6)
    X_travel.to_csv('../data/folks_insurance_FL_cleaned.csv', index=False)

def german_data_cleaning():
    df = pd.read_csv('../data/german.csv', sep='\s+', index_col=False,
                     names=['Account Balance', 'Duration of Credit (month)', 'Payment Status of Previous Credit',
                            'Purpose',
                            'Credit Amount', 'Value Savings/Stocks', 'Length of current employment',
                            'Instalment per cent', 'Sex & Marital Status',
                            'Guarantors', 'Duration in Current address', 'Most valuable available asset', 'Age (years)',
                            'Concurrent Credits', 'Type of apartment', 'No of Credits at this Bank', 'Occupation',
                            'No of dependents',
                            'Telephone', 'Foreign Worker', 'Creditability'])

    # Encoding Categorical Vairable
    def encoding_category(accountBalance):
        return int(accountBalance[-1])

    categorical_columns = [1, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17, 19, 20]
    for i in categorical_columns:
        df.iloc[:, i - 1] = df.iloc[:, i - 1].apply(lambda x: encoding_category(x))

    # Encode Credibility
    def encoding_credit(credibility):
        return credibility - 1

    df['Creditability'] = df['Creditability'].apply(lambda x: encoding_credit(x))
    # Write to a new file
    df.to_csv('../data/german_cleaned.csv', index=False)


folks_insurance_data_cleaning()
folks_income_data_cleaning()
folks_travel_data_cleaning()
folks_mobility_data_cleaning()