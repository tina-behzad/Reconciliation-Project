import math
import random
from csv import DictWriter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
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


class ExperimentThreePipeline:
    def __init__(self, data: Data_Wrapper, model_set, is_classification, alpha, epsilon,
                 model_similarity_threshold):
        self.data = data
        self.model_set = model_set
        self.is_classification = is_classification
        self.alpha = alpha
        self.epsilon = epsilon
        self.model_similarity_threshold = model_similarity_threshold


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

        selected_models = [(name, model) for (name, model) in self.model_set if performance_scores[name] >= threshold]

        M = [ModelWrapper(model,self.data.train_X, self.data.train_y, self.is_classification, None) for (name, model) in selected_models]
        return M
    def write_result_to_file(self,results, method,results_csv_file_name, result_dict):
        result_dict["Method"] = method
        with open(results_csv_file_name, 'a') as f_object:
            for index,key in enumerate(["variance in predictions", "ambiguity","discrepancy","disagreement"]):
                result_dict["Measure"] = key
                result_dict["Value"] = results[index]
                dictwriter_object = DictWriter(f_object, fieldnames=result_dict.keys())
                dictwriter_object.writerow(result_dict)
            f_object.close()

    def get_all_combinations_reconciled(self, set_M):
        model_predictions = []
        for model1, model2 in combinations(set_M, 2):
            reconcile_instance = Reconcile(model1, model2, self.data, self.alpha,
                                       self.epsilon, True)
            u, _, __ = reconcile_instance.find_disagreement_set()
            current_models_disagreement_set_probability_mass = calculate_probability_mass(self.data.get_whole_data(return_test_and_val_only=True), u)

            if current_models_disagreement_set_probability_mass > self.alpha:
                reconcile_instance.reconcile(None, None)
                model1_predictions, model2_predictions = reconcile_instance.get_reconciled_predictions()
                model_predictions.append(self.data.seperate_data_section(model1_predictions, return_section="test"))
                model_predictions.append(self.data.seperate_data_section(model2_predictions, return_section="test"))
                # model1.set_reconcile(model1_predictions)
                # model2.set_reconcile(model2_predictions)
            else:
                print("no need to reconcile")
                model_predictions.append(model1.predict(self.data.test_x))
                model_predictions.append(model2.predict(self.data.test_x))
        return model_predictions

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
        predictions_list = []
        model1 = random.choice(set_M)
        set_M.remove(model1)
        model2 = random.choice(set_M)
        set_M.remove(model2)
        chosen_model, predictions = self.apply_reconcile(model1, model2)
        i = 1
        predictions_list.append(predictions)
        # chosen_model.set_reconcile(chosen_model.predict(X_test.copy().drop(target_col_name, axis=1)))
        while set_M:
            next_model = random.choice(set_M)
            set_M.remove(next_model)
            chosen_model,predictions = self.apply_reconcile(chosen_model, next_model)
            predictions_list.append(predictions)
            # if not chosen_model.reconciled:
            #     chosen_model.set_reconcile(chosen_model.predict(X_test.copy().drop(target_col_name, axis=1)))
        return predictions_list
    def run(self,result_file_name, result_dict):
        set_M = self.create_set_M()
        set_M_predictions = [model.predict(self.data.test_x) for model in set_M]
        results = self.calculate_metrics(set_M_predictions)
        self.write_result_to_file(results,"M",result_file_name,result_dict)

        all_combinations_predictions = self.get_all_combinations_reconciled(set_M)
        results = self.calculate_metrics(all_combinations_predictions)
        self.write_result_to_file(results, "a", result_file_name, result_dict)

        only_one_chosen_after_reconcile_predictions = all_combinations_predictions[0::2]
        results = self.calculate_metrics(only_one_chosen_after_reconcile_predictions)
        self.write_result_to_file(results, "b", result_file_name, result_dict)

        randomly_selected_after_reconcile = random.sample(all_combinations_predictions, len(set_M))
        results = self.calculate_metrics(randomly_selected_after_reconcile)
        self.write_result_to_file(results, "c", result_file_name, result_dict)

        contesting_set_predictions = self.run_sequential_reconcile(set_M)
        results = self.calculate_metrics(contesting_set_predictions)
        self.write_result_to_file(results, "d", result_file_name, result_dict)

    def calculate_ambiguity(self, predictions):
        max_abs_diff_per_point = predictions.apply(lambda x: x.max() - x.min(), axis=1)
        total_diff = max_abs_diff_per_point.sum() / len(max_abs_diff_per_point)
        return total_diff

    def calculate_discrepancy(self, predictions):
        max_diff = -math.inf
        # Generate all combinations of column pairs
        for col1, col2 in combinations(predictions.columns, 2):
            diff_summ = np.sum(abs(predictions[col1] - predictions[col2])) / predictions.shape[0]
            if diff_summ > max_diff:
                max_diff = diff_summ

        return max_diff

    def calculate_disagreement(self, predictions):
        disagreements = []
        # Generate all combinations of column pairs
        for col1, col2 in combinations(predictions.columns, 2):
            diff = (predictions[col1] - predictions[col2]).abs()
            u_epsilon = predictions[diff > self.epsilon]
            disagreements.append(calculate_probability_mass(self.data.get_whole_data(return_test_and_val_only=True), u_epsilon))

        return disagreements

    def calculate_metrics(self, prediction_lists):
        predictions = pd.DataFrame.from_dict(
            {"model_" + str(i): prediction_lists[i] for i in range(0, len(prediction_lists))})
        variance_in_perdictions = predictions.var(axis=1)
        ambiguity = self.calculate_ambiguity(predictions)
        # logging.info(f"ambiguity for the set over predictions is {ambiguity:.4f}")
        discrepency = self.calculate_discrepancy(predictions)
        # logging.info(f"discrepancy for the set over predictions is {discrepency:.4f}")
        # logging.info("Disagreement values")
        # logging.info(stats.describe(self.calculate_disagreement(predictions)))
        return [stats.describe(variance_in_perdictions), round(ambiguity,3),round(discrepency,3),stats.describe(self.calculate_disagreement(predictions))]

