import random
from csv import DictWriter
from itertools import combinations

import pandas as pd
from sklearn._loss import loss
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reconcile import Reconcile
from utils import calculate_probability_mass
from wrappers.data_wrapper import Data_Wrapper
from wrappers.model_aggregator import ModelAggregator
from wrappers.model_wrapper import ModelWrapper


class ExperimentTwoPipeline:
    def __init__(self, data: Data_Wrapper, model_set, is_classification, alpha, epsilon,
                 model_similarity_threshold):
        self.data = data
        self.model_set = model_set
        self.is_classification = is_classification
        self.alpha = alpha
        self.epsilon = epsilon
        self.model_similarity_threshold = model_similarity_threshold
        self.techniques = ['mode', 'randomized', 'random_selection'] if self.is_classification else ['mean', 'randomized', 'random_selection']


    def create_set_M(self):
        performance_scores = {}
        for name, model in self.model_set:
            scaled_model = Pipeline([
                ('normalizer', StandardScaler()),
                ('classifier', model)])
            scoring = "accuracy" if self.is_classification else "neg_mean_squared_error"
            scores = cross_val_score(scaled_model, self.data.train_X, self.data.train_y, cv=5, scoring=scoring)
            performance_scores[name] = scores.mean()
            print("{} is done".format(name))
        top_score = max(performance_scores.values())
        threshold = top_score - self.model_similarity_threshold

        selected_models = [(name, model) for (name, model) in self.model_set if performance_scores[name] >= threshold]

        M = [ModelWrapper(model,self.data.train_X, self.data.train_y, self.is_classification, None) for (name, model) in selected_models]
        return M
    def write_result_to_file(self,score, method,results_csv_file_name, result_dict):
        result_dict["Method"], result_dict["MSE"] = method, round(score, 3)
        with open(results_csv_file_name, 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=result_dict.keys())
            dictwriter_object.writerow(result_dict)
            f_object.close()
    def run(self,result_file_name, result_dict):
        set_M = self.create_set_M()
        for technique in self.techniques:
            model_aggregator = ModelAggregator(set_M, technique=technique)
            predictions = model_aggregator.predict(self.data)
            score = mean_squared_error(self.data.test_y,predictions)
            self.write_result_to_file(score, technique, result_file_name, result_dict)
        reconcile_predictions = self.run_sequential_reconcile(set_M)
        score = mean_squared_error(self.data.test_y, reconcile_predictions)
        self.write_result_to_file(score, "sequential reconcile", result_file_name, result_dict)

    def apply_reconcile(self,model1, model2):
        reconcile_instance = Reconcile(model1, model2, self.data, self.alpha,
                                       self.epsilon, True)
        u, _, __ = reconcile_instance.find_disagreement_set()
        current_models_disagreement_set_probability_mass = calculate_probability_mass(self.data.get_whole_data(return_test_and_val_only=True), u)
        if current_models_disagreement_set_probability_mass > self.alpha:
            chosen_model,complete_chosen_predictions = reconcile_instance.reconcile(None,None)
            model1_predictions, model2_predictions = reconcile_instance.get_reconciled_predictions()
            model1.set_reconcile(model1_predictions)
            model2.set_reconcile(model2_predictions)
        else:
            chosen_model_subscript = random.choice([1, 2])
            chosen_model = model1 if chosen_model_subscript == 1 else model2
            prediction1, prediction2 = reconcile_instance.get_reconciled_predictions()
            complete_chosen_predictions = prediction1 if chosen_model_subscript == 1 else prediction2
        return chosen_model, self.data.seperate_data_section(complete_chosen_predictions, return_section="test")

    def run_sequential_reconcile(self, set_M):
        prediction_history = pd.DataFrame(columns=[f'model{i}_prediction' for i in range(1, len(set_M) + 1)], index = self.data.test_x.index)
        model1 = random.choice(set_M)
        set_M.remove(model1)
        model2 = random.choice(set_M)
        set_M.remove(model2)
        chosen_model, predictions = self.apply_reconcile(model1, model2)
        i = 1
        prediction_history[f'model{i}_prediction'] = predictions
        # chosen_model.set_reconcile(chosen_model.predict(X_test.copy().drop(target_col_name, axis=1)))
        while set_M:
            next_model = random.choice(set_M)
            set_M.remove(next_model)
            chosen_model,predictions = self.apply_reconcile(chosen_model, next_model)
            i+=1
            prediction_history[f'model{i}_prediction'] = predictions
            # if not chosen_model.reconciled:
            #     chosen_model.set_reconcile(chosen_model.predict(X_test.copy().drop(target_col_name, axis=1)))
        return prediction_history[f'model{i}_prediction']




