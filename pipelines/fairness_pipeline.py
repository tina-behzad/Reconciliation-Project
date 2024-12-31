import copy
import random
from csv import DictWriter
from itertools import combinations

import pandas as pd
from sklearn._loss import loss
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from wrappers.data_wrapper import Data_Wrapper
from wrappers.model_aggregator import ModelAggregator
from wrappers.model_wrapper import ModelWrapper
from pipelines.experiment2_pipeline import ExperimentTwoPipeline



class Fairness_Pipeline(ExperimentTwoPipeline):

    def run(self,result_file_name, result_dict):
        set_M = self.create_set_M()
        self.calculate_original_MSE_for_groups(set_M, result_file_name, result_dict)
        for technique in self.techniques:
            model_aggregator = ModelAggregator(set_M, technique=technique)
            predictions = model_aggregator.predict(self.data)
            self.calculate_MSE_for_groups(predictions, technique,result_file_name, result_dict)
        reconcile_predictions = self.run_sequential_reconcile(set_M)
        self.calculate_MSE_for_groups(reconcile_predictions, "reconcile", result_file_name, result_dict)


    def write_result_to_file(self,scores, method,results_csv_file_name, result_dict):
        result_dict["MSE for different groups"], result_dict["Method"] = scores,method
        with open(results_csv_file_name, 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=result_dict.keys())
            dictwriter_object.writerow(result_dict)
            f_object.close()
    def calculate_original_MSE_for_groups(self, set_M,results_csv_file_name, result_dict):
        group_datasets = self.data.get_groups_data()
        true_labels = [self.data.return_intersection_true_label(group_data, "test") for group_data in group_datasets]
        MSE_list = []
        for model in set_M:
            predictions = [model.predict(data) for data in group_datasets]
            MSE_list.append([mean_squared_error(true_label,prediction) for (true_label,prediction) in zip(true_labels,predictions)])
        self.write_result_to_file(MSE_list,"original",results_csv_file_name, result_dict)


    def calculate_MSE_for_groups(self, model_predictions,method, results_csv_file_name, result_dict):
        group_datasets = self.data.get_groups_data()
        aligned_predictions = pd.Series(model_predictions, index=self.data.test_y.index)
        true_labels = [self.data.return_intersection_true_label(group_data, "test") for group_data in group_datasets]
        predictions = [aligned_predictions[aligned_predictions.index.intersection(group_data.index)] for group_data in group_datasets]
        MSE_list = [mean_squared_error(true_label,prediction) for (true_label,prediction) in zip(true_labels,predictions)]
        self.write_result_to_file(MSE_list, method, results_csv_file_name, result_dict)
