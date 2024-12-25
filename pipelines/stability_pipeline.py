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



class Stability_Pipeline(ExperimentTwoPipeline):
    def __init__(self, data: Data_Wrapper, model_set, is_classification, alpha, epsilon,
                 model_similarity_threshold, dataset_set_size = None):
        super().__init__(data, model_set, is_classification, alpha, epsilon,model_similarity_threshold)
        self.dataset_set_size = dataset_set_size

    def create_set_M(self):
        performance_scores = {}
        for name, model in self.model_set:
            scaled_model = Pipeline([
                ('normalizer', StandardScaler()),
                ('classifier', model)])
            scoring = "accuracy" if self.is_classification else "neg_mean_squared_error"
            scores = cross_val_score(scaled_model, self.data.train_X, self.data.train_y, cv=5, scoring=scoring)
            performance_scores[name] = scores.mean()
            # print("{} is done".format(name))
        top_score = max(performance_scores.values())
        threshold = top_score - self.model_similarity_threshold

        if self.dataset_set_size is None:
            selected_models = [(name, model) for (name, model) in self.model_set if performance_scores[name] >= threshold]
            self.dataset_set_size = len(selected_models)
        else:
            top_keys = [key for key, value in sorted(performance_scores.items(), key=lambda item: item[1], reverse=True)[:self.dataset_set_size]]
            selected_models = [(name, model) for (name, model) in self.model_set if name in top_keys]
        M = [ModelWrapper(model,self.data.train_X, self.data.train_y, self.is_classification, None) for (name, model) in selected_models]
        return M



    def run(self,result_file_name, result_dict):
        set_M = self.create_set_M()
        result_dict["Set Size"] = len(set_M)
        replaced_indices = set()
        for number_of_random_models in range(self.dataset_set_size+1):
            if number_of_random_models != 0:
                available_indices = [i for i in range(len(set_M)) if i not in replaced_indices]
                random_index = random.choice(available_indices)
                set_M[random_index] = ModelWrapper("Dummy",self.data.train_X, self.data.train_y, self.is_classification, None)
                replaced_indices.add(random_index)
            individual_mse_values = [round(model.get_brier_score(self.data.test_x,self.data.test_y),4) for model in set_M]
            result_dict["Individual MSE"] = individual_mse_values
            result_dict["number of random models"] = number_of_random_models
            for technique in self.techniques:
                model_aggregator = ModelAggregator(set_M, technique=technique)
                predictions = model_aggregator.predict(self.data)
                score = mean_squared_error(self.data.test_y,predictions)
                self.write_result_to_file(score, technique, result_file_name, result_dict)
            reconcile_predictions = self.run_sequential_reconcile(copy.deepcopy(set_M))
            score = mean_squared_error(self.data.test_y, reconcile_predictions)
            self.write_result_to_file(score, "sequential reconcile", result_file_name, result_dict)
        return self.dataset_set_size
