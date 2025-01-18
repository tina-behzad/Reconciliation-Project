import copy
import random
from csv import DictWriter
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn._loss import loss
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reconcile import Reconcile
from wrappers.data_wrapper import Data_Wrapper
from wrappers.model_aggregator import ModelAggregator
from wrappers.model_wrapper import ModelWrapper
from pipelines.experiment2_pipeline import ExperimentTwoPipeline



class Fairness_Pipeline(ExperimentTwoPipeline):

    def run(self,result_file_name, result_dict, set_or_pair = 'set'):
        if set_or_pair == 'set':
            set_M = super().create_set_M()
            self.calculate_original_MSE_for_groups(set_M, result_file_name, result_dict)
            for technique in self.techniques:
                model_aggregator = ModelAggregator(set_M, technique=technique)
                predictions = model_aggregator.predict(self.data)
                self.calculate_MSE_for_groups(predictions, technique,result_file_name, result_dict)
            reconcile_predictions = self.run_sequential_reconcile(set_M)
            self.calculate_MSE_for_groups(reconcile_predictions, "reconcile", result_file_name, result_dict)
        elif set_or_pair == 'pair':
            #choose only the best 2 models from the set
            model_pairs = self.create_set_M()
            if(len(model_pairs)!=2):
                print("couldn't find models within the threshold!")
                return
            disrupted_group_index = self.disrupt_group_predictions(model_pairs[1])
            self.calculate_original_MSE_for_pair(model_pairs,disrupted_group_index,result_file_name,result_dict)
            reconcile_instance = Reconcile(model_pairs[0],model_pairs[1],self.data, self.alpha,self.epsilon)
            reconcile_instance.reconcile(None,None, False)
            first_model_predictions, second_model_predictions = reconcile_instance.get_reconciled_predictions()
            self.calculate_final_MSE_for_pair(first_model_predictions,second_model_predictions,disrupted_group_index, result_file_name, result_dict)

    def create_set_M(self, same_model = False):
        performance_scores = {}
        selected_models = []
        if same_model:
            return [ModelWrapper(self.model_set[0][1], self.data.train_X, self.data.train_y, self.is_classification, None),
                    ModelWrapper(self.model_set[0][1], self.data.train_X, self.data.train_y, self.is_classification, None)]
        for name, model in self.model_set:
            # Train the current model and evaluate its performance
            scaled_model = Pipeline([
                ('normalizer', StandardScaler()),
                ('classifier', model)
            ])
            scoring = "accuracy" if self.is_classification else "neg_mean_squared_error"
            scores = cross_val_score(scaled_model, self.data.train_X, self.data.train_y, cv=5, scoring=scoring)
            performance_scores[name] = scores.mean()

            # Identify the threshold dynamically
            if len(performance_scores) == 1:
                top_score = scores.mean()  # First model sets the top score
                threshold = top_score - self.model_similarity_threshold

            # Check if the current model meets the threshold
            if performance_scores[name] >= threshold:
                selected_models.append((name, model))

            # Exit early if two models are found
            if len(selected_models) == 2:
                break

        # Wrap selected models
        M = [ModelWrapper(model, self.data.train_X, self.data.train_y, self.is_classification, None) for (name, model)
             in selected_models]
        print(round(mean_squared_error(self.data.test_y,M[0].predict(self.data.test_x)) - mean_squared_error(self.data.test_y,M[1].predict(self.data.test_x)),4))
        return M


    def write_result_to_file(self,scores, method,ratio,results_csv_file_name, result_dict):
        result_dict["Method"], result_dict["MSE"], result_dict["ratio"] = method,scores,ratio
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


    def disrupt_group_predictions(self, model:ModelWrapper):
        group_datasets = self.data.get_groups_data(return_section='test_val')
        sizes = [(i, len(df)) for i, df in enumerate(group_datasets)]  # List of (index, size)
        sorted_sizes = sorted(sizes, key=lambda x: x[1], reverse=True)
        second_largest_index = sorted_sizes[1][0]
        second_largest_df = group_datasets[second_largest_index]
        # if self.is_classification:
        #     random_values = np.random.randint(0, 2, size=len(second_largest_df))
        # else:
        random_values = np.random.uniform(0, 1, size=len(second_largest_df))

        predictions_for_group = pd.Series(random_values, index=second_largest_df.index)
        complete_predictions = [
            pd.Series(model.predict(data), index=data.index, name='Prediction')
            for data in group_datasets
        ]
        complete_predictions[second_largest_index] = predictions_for_group
        predictions = pd.concat(complete_predictions)
        model.set_reconcile(predictions)
        return second_largest_index

    def calculate_original_MSE_for_pair(self, models,group_index,results_csv_file_name, result_dict):
        #TODO should I not return test and val here as well?
        group_datasets = self.data.get_groups_data()
        minority_data = group_datasets[group_index]
        test_true_labels = self.data.test_y
        majority_labels = test_true_labels.drop(minority_data.index)
        minority_labels = test_true_labels[test_true_labels.index.intersection(minority_data.index)]
        MSE_list = []
        first_prediction_majority = pd.Series(models[0].predict(self.data.test_x), index=self.data.test_y.index).drop(minority_data.index)
        first_prediction_minority = pd.Series(models[0].predict(minority_data), index = minority_data.index)
        MSE_list.append([mean_squared_error(majority_labels,first_prediction_majority), mean_squared_error(minority_labels,first_prediction_minority)])
        #second model's is_reconciled is set when predictions are disrupted so the returned predictions include validation
        second_predictions = self.data.seperate_data_section(models[1].predict(self.data.test_x),'test')
        second_prediction_majority = second_predictions.drop(minority_data.index)
        second_prediction_minority = second_predictions[second_predictions.index.intersection(minority_data.index)]
        MSE_list.append([mean_squared_error(majority_labels, second_prediction_majority),
                         mean_squared_error(minority_labels, second_prediction_minority)])
        self.write_result_to_file(MSE_list,"original",[len(majority_labels),len(minority_labels)],results_csv_file_name, result_dict)

    def calculate_final_MSE_for_pair(self, first_model_predictions, second_model_predictions, disrupted_group_index, results_csv_file_name, result_dict):
        group_datasets = self.data.get_groups_data()
        minority_data = group_datasets[disrupted_group_index]
        test_true_labels = self.data.seperate_data_section(self.data.get_all_labels(return_test_and_val_only=True),
                                                           return_section='test')
        majority_labels = test_true_labels.drop(minority_data.index)
        minority_labels = test_true_labels[test_true_labels.index.intersection(minority_data.index)]
        MSE_list = []
        for prediction in [first_model_predictions,second_model_predictions]:

            test_only_predictions = self.data.seperate_data_section(prediction, 'test')
            second_prediction_majority = test_only_predictions.drop(minority_data.index)
            second_prediction_minority = test_only_predictions[test_only_predictions.index.intersection(minority_data.index)]
            MSE_list.append([mean_squared_error(majority_labels, second_prediction_majority),
                             mean_squared_error(minority_labels, second_prediction_minority)])
        self.write_result_to_file(MSE_list, "Reconciled",[len(majority_labels),len(minority_labels)], results_csv_file_name, result_dict)