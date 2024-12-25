import ast
import configparser
import os
from datetime import datetime

import pandas as pd
import numpy as np
import itertools
from IPython.display import Image
from tqdm import tqdm
from xgboost import XGBRegressor
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor
from causalml.dataset import make_uplift_classification
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier, CausalTreeRegressor
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot
from sklearn.model_selection import train_test_split
from causalml.inference.tree import CausalRandomForestRegressor
import traceback
from reconcile import Reconcile
from wrappers.CATE_estimator import CATE_estimator
from wrappers.causal_data_wrapper import CausalDataWrapper
from wrappers.data_wrapper import Data_Wrapper
from wrappers.model_wrapper import ModelWrapper


def Reconcile_CATE(estimator1, estimator2, test_data, outcome_column, treatment_column, treatment_values,
                   results_csv_file_name, result_dict, alpha, epsilon):
    group_errors = []
    for group_number in test_data['leaf_number'].unique():
        X_group = test_data[test_data['leaf_number'] == group_number].drop(
            columns=['leaf_number', outcome_column, treatment_column],
            axis=1)
        predicted_cate_1 = estimator1.predict(X=X_group.values).mean()
        predicted_cate_2 = estimator2.predict(X=X_group.values).mean()
        actual_cate = \
        test_data[(test_data['leaf_number'] == group_number) & (test_data[treatment_column] == treatment_values[1])][
            outcome_column].mean() - \
        test_data[(test_data['leaf_number'] == group_number) & (test_data[treatment_column] == treatment_values[0])][
            outcome_column].mean()

        group_errors.append({
            'group': group_number,
            'actual_cate': actual_cate,
            'predicted_1': predicted_cate_1,
            'predicted_2': predicted_cate_2
        })
    group_df = pd.DataFrame(group_errors)
    data = CausalDataWrapper(group_df, 'actual_cate', 'group')
    model1 = ModelWrapper(None, None, group_df['predicted_1'], None, None, True)
    model2 = ModelWrapper(None, None, group_df['predicted_2'], None, None, True)
    # result_dict = {"Data": "Twin", "Method": "Different_Data", "Models": 'uplift tree'}
    reconcile_instance = Reconcile(model1, model2, data, alpha, epsilon)
    reconcile_instance.reconcile(results_csv_file_name, result_dict)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('configs.ini')
    results_csv_file_name = "./results/CATE  " + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    columns = ["Data", "Method", "Models", "Initial Disagreement", "Final Disagreement", "Initial Brier", "Final Brier",
               "Theorem 3.1", "T1", "T2", "Brier Scores"]
    results_df = pd.DataFrame(
        columns=columns).to_csv(results_csv_file_name, index=False)
    alpha = float(config['CATE_Configs']['Alpha'])
    epsilon = float(config['CATE_Configs']['Epsilon'])
    Experiment_Repeat_Numbers = int(config['CATE_Configs']['Experiment_Repeat_Numbers'])
    for i in tqdm(range(Experiment_Repeat_Numbers)):
        models = ast.literal_eval(config["CATE_Configs"]['Estimators'])
        for dataset_name in ast.literal_eval(config['CATE_Configs']['Datasets']):
            data_path = os.getcwd() + config[dataset_name]['Address']
            outcome_column = config[dataset_name]['Outcome_Column_name']
            treatment_column = config[dataset_name]['Treatment_Column_name']
            control_values = ast.literal_eval(config[dataset_name]['Control_Group_Value'])
            data = pd.read_csv(data_path, dtype={treatment_column:int})
            for method in ast.literal_eval(config['CATE_Configs']['Possible_approaches']):
                if method == 'different data':
                    train_1_df = []
                    train_2_df = []
                    test_dfs = []
                    for group in data['leaf_number'].unique():
                        df_group = data[data['leaf_number'] == group]
                        train_group, test_group = train_test_split(df_group, train_size=0.5)
                        train_1, train_2 = train_test_split(train_group, train_size=0.5)
                        train_1_df.append(train_1)
                        train_2_df.append(train_2)
                        test_dfs.append(test_group)

                    # Concatenate all group train/test splits
                    df_train_1 = pd.concat(train_1_df).reset_index(drop=True)
                    df_train_2 = pd.concat(train_2_df).reset_index(drop=True)
                    df_test = pd.concat(test_dfs).reset_index(drop=True)
                    for estimator_name in models:
                        estimator1 = CATE_estimator(estimator_name)
                        estimator1.fit(
                            X=df_train_1.drop(columns=[treatment_column, 'leaf_number', outcome_column], axis=1).values,
                            treatment=df_train_1[treatment_column].to_numpy(), y=df_train_1[outcome_column].to_numpy())
                        estimator2 = CATE_estimator(estimator_name)
                        estimator2.fit(
                            X=df_train_2.drop(columns=[treatment_column, 'leaf_number', outcome_column], axis=1).values,
                            treatment=df_train_2[treatment_column].to_numpy(), y=df_train_2[outcome_column].to_numpy())
                        result_dict = {"Data": dataset_name, "Method": method, "Models": estimator_name}
                        try:
                            Reconcile_CATE(estimator1, estimator2, df_test, outcome_column, treatment_column,
                                           control_values, results_csv_file_name, result_dict, alpha, epsilon)
                        except Exception as e:
                            print("reconcile failed for {}".format(result_dict))
                            traceback.print_exc()

                elif method == 'different models':
                    test_dfs = []
                    train = []
                    for group in data['leaf_number'].unique():
                        df_group = data[data['leaf_number'] == group]
                        train_group, test_group = train_test_split(df_group, train_size=0.5)
                        # Collect the splits
                        train.append(train_group)
                        test_dfs.append(test_group)
                    df_test = pd.concat(test_dfs).reset_index(drop=True)
                    df_train = pd.concat(train).reset_index(drop=True)
                    for estimator1_name, estimator2_name in itertools.combinations(
                            models, 2):
                        estimator1 = CATE_estimator(estimator1_name)
                        estimator1.fit(
                            X=df_train.drop(columns=[treatment_column, 'leaf_number', outcome_column], axis=1).values,
                            treatment=df_train[treatment_column].to_numpy(), y=df_train[outcome_column].to_numpy())
                        estimator2 = CATE_estimator(estimator2_name)
                        estimator2.fit(
                            X=df_train.drop(columns=[treatment_column, 'leaf_number', outcome_column], axis=1).values,
                            treatment=df_train[treatment_column].to_numpy(), y=df_train[outcome_column].to_numpy())
                        result_dict = {"Data": dataset_name, "Method": method,
                                       "Models": [estimator1_name, estimator2_name]}
                        try:
                            Reconcile_CATE(estimator1, estimator2, df_test, outcome_column, treatment_column,
                                           control_values, results_csv_file_name, result_dict, alpha, epsilon)
                        except Exception as e:
                            print("reconcile failed for {}".format(result_dict))
                            traceback.print_exc()
