import itertools
import random
from csv import DictWriter

import numpy as np
import pandas as pd
import math
from enum import Enum
from utils import calculate_probability_mass, round_to_fraction, create_log_file_name
import logging

from wrappers.data_wrapper import Data_Wrapper
from wrappers.model_wrapper import ModelWrapper
from datetime import datetime
import matplotlib.pyplot as plt


class Subscript(Enum):
    greater = 0
    smaller = 1


class Reconcile:
    def __init__(self, f1: ModelWrapper, f2: ModelWrapper, dataset: Data_Wrapper, alpha: float, epsilon: float, exp_2_3 = False ):
        """
        Initializes the Reconcile class with two models, a dataset, and parameters alpha and epsilon.

        Parameters:
        f1: The first model (it should be an instance of the ModelWrapper class).
        f2: The second model (it should be an instance of the ModelWrapper class).
        dataset: The dataset to be used(In form of Data_Wrapper class).
        alpha: Approximate group conditional mean consistency parameter.
        epsilon: disagreement threshhold.
        """
        self.model1 = f1
        self.model2 = f2
        self.data = dataset
        self.is_experiment_2_3 = exp_2_3
        # self.dataset.insert(0, 'assigned_id', range(0, len(self.dataset)))
        # self.target_feature = target_feature_name
        self.alpha = alpha
        self.epsilon = epsilon
        # self.trained_on_different_features = trained_on_different_features
        # self.model_feature_lists = model_feature_lists
        self.predicitons_history_df = pd.DataFrame(columns=['f1_predictions', 'f2_predictions'],
                                                   index=self.data.get_whole_data(return_test_and_val_only=True).index)
        self.predicitons_history_df["actual_labels"] = self.data.get_all_labels(return_test_and_val_only = True)
        self.predicitons_history_df['f1_predictions'], self.predicitons_history_df[
            'f2_predictions'] = self.get_model_predictions()
        logging.basicConfig(filename='./logs/' + create_log_file_name(self.alpha, self.epsilon) + ".log",
                            encoding='utf-8',
                            level=logging.DEBUG, format='%(asctime)s %(message)s')
        logging.getLogger('matplotlib.font_manager').disabled = True
        logging.info(
            f'model 1 name: {type(self.model1.model[1]).__name__} model 2 name: {type(self.model2.model[1]).__name__}')
        # logging.info(f'dataset name: {dataset_name}')

    def get_model_predictions(self):
        """
        Use models f1 and f2 to make predictions on the dataset.
        Add these predictions as new columns to the dataset.
        """
        f1_predictions = self.model1.predict(self.data)
        f2_predictions = self.model2.predict(self.data)
        return f1_predictions, f2_predictions

    def find_disagreement_set(self):
        """
                Find sets of data points where the predictions of f1 and f2 differ significantly.
        """
        # if 'f1_predictions' not in self.dataset.columns or 'f2_predictions' not in self.dataset.columns:
        #     self.get_model_predictions()
        if self.predicitons_history_df.empty:
            f1_predictions, f2_predictions = self.get_model_predictions()
        else:
            # get latest predictions for f1
            f1_predictions = self.predicitons_history_df[
                [col for col in self.predicitons_history_df.columns if 'f1' in col][-1]]
            f2_predictions = self.predicitons_history_df[
                [col for col in self.predicitons_history_df.columns if 'f2' in col][-1]]
        diff = (f1_predictions - f2_predictions).abs()
        u_epsilon = self.data.get_whole_data(return_test_and_val_only=True)[diff > self.epsilon]
        u_epsilon_greater = u_epsilon[f1_predictions > f2_predictions]
        u_epsilon_smaller = u_epsilon[f1_predictions < f2_predictions]
        return u_epsilon, u_epsilon_greater, u_epsilon_smaller

    def calculate_consistency_violation(self, u, v_star, v):
        return calculate_probability_mass(self.data.get_whole_data(return_test_and_val_only=True), u) * pow(
            (v_star - v), 2)

    def find_candidate_for_update(self, u_greater, u_smaller):
        u = [u_greater, u_smaller]
        v_star = [self.data.return_intersection_true_label(u_greater, return_section="val").mean(),
                  self.data.return_intersection_true_label(u_smaller, return_section="val").mean()]
        f1_predictions = self.predicitons_history_df[
            [col for col in self.predicitons_history_df.columns if 'f1' in col][-1]]
        f2_predictions = self.predicitons_history_df[
            [col for col in self.predicitons_history_df.columns if 'f2' in col][-1]]
        v = [[self.data.seperate_intersection_data(f1_predictions, u_greater, return_section="test_val").mean(),
              self.data.seperate_intersection_data(f1_predictions, u_smaller, return_section="test_val").mean()],
             [self.data.seperate_intersection_data(f2_predictions, u_greater, return_section="test_val").mean(),
              self.data.seperate_intersection_data(f2_predictions, u_smaller, return_section="test_val").mean()]
             ]
        consistency_violation = -math.inf
        selected_subscript = -1
        selected_i = -1
        selected_candidates = []
        for subscript, i in itertools.product([Subscript.greater.value, Subscript.smaller.value], [0, 1]):

            new_consistency_violation = self.calculate_consistency_violation(u[subscript], v_star[subscript],
                                                                             v[i][subscript])
            if new_consistency_violation > consistency_violation:
                consistency_violation = new_consistency_violation
                selected_candidates.clear()
                selected_candidates.append([subscript, i])
            elif new_consistency_violation == consistency_violation:
                selected_candidates.append([subscript, i])

        # Break the tie arbitrary
        selected_candidate = random.choice(selected_candidates)
        selected_subscript = selected_candidate[0]
        selected_i = selected_candidate[1]
        return selected_subscript, selected_i

    def patch(self, predictions, g, delta):
        # subset_ids = set(g['assigned_id'])
        predictions.loc[predictions.index.isin(g.index)] += delta
        # apply project
        predictions = predictions.clip(lower=0, upper=1)
        return predictions

    def log_round_info(self, t, u, chosen_subscript, index, delta):
        message = (
            f'round {t} complete. Miu(u) = {round(calculate_probability_mass(self.data.get_whole_data(return_test_and_val_only=True), u), 4)}.'
            f' u {Subscript(chosen_subscript).name} f_{index + 1} was updated. update value(delta) = {delta}')
        logging.info(message)

    def final_round_logs(self, brier_scores, u, t, t1, t2, time, initial_disagreement, result_file_name, result_dict):
        rounded_brier_scores = [[round(score[0], 3), round(score[1], 3)] for score in brier_scores]
        message = (f'total rounds {t} completed in {time}. T1 = {t1} T2={t2} \n'
                   f'brier_scores = {rounded_brier_scores} \n')
        theorem31, final_disagreement = self.calculate_theorem31(t, t1, t2, rounded_brier_scores[0],
                                                                 rounded_brier_scores[-1], u)
        result_dict["Initial Disagreement"], result_dict["Final Disagreement"] = round(initial_disagreement,
                                                                                       3), final_disagreement
        result_dict["Initial Brier"], result_dict["Final Brier"] = rounded_brier_scores[0], rounded_brier_scores[-1]
        result_dict["Theorem 3.1"], result_dict["T1"], result_dict["T2"] = theorem31, t1, t2
        result_dict["Brier Scores"] = rounded_brier_scores
        with open(result_file_name, 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=result_dict.keys())
            dictwriter_object.writerow(result_dict)
            f_object.close()
        print(message)
        logging.info(message)
        logging.info(theorem31)

    def calculate_theorem31(self, t, t1, t2, initial_brier_scores, final_brier_scores, u):
        multiplier = (16 / (self.alpha * pow(self.epsilon, 2)))
        rounds_upper_limit = round((final_brier_scores[0] + final_brier_scores[1]) * multiplier, 3)
        brier_score_update_f1 = round(initial_brier_scores[0] - t1 * (1 / multiplier), 3)
        brier_score_update_f2 = round(initial_brier_scores[1] - t2 * (1 / multiplier), 3)
        final_mu = round(calculate_probability_mass(self.data.test_x,
                                                          self.data.seperate_data_section(u, return_section="test")), 3)
        message = f'1. {t} <= {rounds_upper_limit} |2. {final_brier_scores[0]} <= {brier_score_update_f1} and {final_brier_scores[1]} <= {brier_score_update_f2} |3. {final_mu} < {self.alpha}'
        print(message)
        return message, final_mu

    def reconcile(self, result_file_name, result_dict):
        start_time = datetime.now()
        t = 0
        t1 = 0
        t2 = 0
        m = round(2 / (math.sqrt(self.alpha) * self.epsilon))
        brier_scores = [
            [self.model1.get_brier_score(
                self.data.seperate_data_section(self.predicitons_history_df['f1_predictions'], return_section="test"),
                self.data.test_y, True),
                self.model2.get_brier_score(
                    self.data.seperate_data_section(self.predicitons_history_df['f2_predictions'],
                                                    return_section="test"),
                    self.data.test_y, True)]]
        u, u_greater, u_smaller = self.find_disagreement_set()
        # TODO should probability mass be over the whole dataset?
        #  Decided: for reports, only test for calculations both test and val!
        initial_disagreement = calculate_probability_mass(self.data.test_x,
                                                          self.data.seperate_data_section(u, return_section="test"))
        print("initial disagreement level = {}".format(initial_disagreement))
        logging.info("initial disagreement level = {}".format(initial_disagreement))
        while calculate_probability_mass(self.data.get_whole_data(return_test_and_val_only=True), u) >= self.alpha:
            subscript, i = self.find_candidate_for_update(u_greater, u_smaller)
            # selected_model = self.model1 if i==0 else self.model2
            selected_model_predictions_col_name = "f1_predictions" if i == 0 else "f2_predictions"
            selected_model_predictions = self.predicitons_history_df[
                [col for col in self.predicitons_history_df.columns if selected_model_predictions_col_name in col][-1]]
            g = u_greater if subscript == Subscript.greater.value else u_smaller
            if self.model1.return_probs:
                predictions_for_delta = self.data.seperate_intersection_data(selected_model_predictions, g, return_section="test_val")
                predictions_for_delta = np.where(predictions_for_delta >= 0.5,1,0).mean()
            else:
                predictions_for_delta = self.data.seperate_intersection_data(
                selected_model_predictions, g, return_section="test_val").mean()
            delta = self.data.return_intersection_true_label(g,
                                                             return_section="val").mean() - predictions_for_delta
            delta = round_to_fraction(delta, m)
            new_predictions = self.patch(selected_model_predictions.copy(), g, delta)
            self.log_round_info(t, u, subscript, i, delta)
            self.predicitons_history_df[f'{t}_{selected_model_predictions_col_name}'] = new_predictions
            f1_latest_prediction, f2_latest_prediction = self.get_reconciled_predictions()
            brier_scores.append(
                [self.model1.get_brier_score(
                    self.data.seperate_data_section(f1_latest_prediction, return_section="test"), self.data.test_y,
                    True),
                 self.model2.get_brier_score(
                     self.data.seperate_data_section(f2_latest_prediction, return_section="test"), self.data.test_y,
                     True)])
            if i == 0:
                t1 += 1
            else:
                t2 += 1
            t += 1
            u, u_greater, u_smaller = self.find_disagreement_set()

        end_time = datetime.now()
        if self.is_experiment_2_3:
            chosen_model_subscript = random.choice([1,2])
            chosen_model = self.model1 if chosen_model_subscript == 1 else self.model2
            prediction1, prediction2 = self.get_reconciled_predictions()
            chosen_model_predictions = prediction1 if chosen_model_subscript == 1 else prediction2

            return chosen_model, chosen_model_predictions
        self.final_round_logs(brier_scores, u, t, t1, t2, (end_time - start_time).seconds, initial_disagreement,
                              result_file_name, result_dict)
        self.predicitons_history_df.to_csv('./logs/datasets/new/'+ result_dict["Data"] + result_dict["Method"]+ result_dict["Models"] + create_log_file_name(self.alpha, self.epsilon) + ".csv",
                                           index=False)
        return brier_scores[-1]
        # self.plot(brier_scores, t)

    def get_reconciled_predictions(self):
        f1_final_column_name = [col for col in self.predicitons_history_df.columns if 'f1' in col][-1]
        f2_final_column_name = [col for col in self.predicitons_history_df.columns if 'f2' in col][-1]
        return self.predicitons_history_df[f1_final_column_name], self.predicitons_history_df[f2_final_column_name]

    def plot(self, brier_scores, t):
        model1_scores = [sublist[0] for sublist in brier_scores]
        model2_scores = [sublist[1] for sublist in brier_scores]
        t_values = list(range(0, t + 1))
        plt.figure(figsize=(10, 6))  # Set the figure size for better visibility
        plt.plot(t_values, model1_scores, 'ro-', label='First Model', linewidth=2,
                 markersize=8)  # Red for the first elements
        plt.plot(t_values, model2_scores, 'bo-', label='Second Model', linewidth=2,
                 markersize=8)  # Blue for the second elements

        plt.title('MSE score over rounds', fontsize=16)  # Title of the plot
        plt.xlabel('round number', fontsize=14)  # X-axis label
        plt.ylabel('MSE', fontsize=14)  # Y-axis label
        plt.xticks(t_values)  # Ensure x-axis ticks match t values for clarity
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add a grid for easier reading
        plt.legend(fontsize=12)  # Add a legend to distinguish between first and second elements

        plt.tight_layout()  # Adjust the layout to make room for the elements
        plt.show()  # Display the plot

    # def flush_and_reset_prediction_history(self,model1_prediction, model2_prediction):
    #     self.predicitons_history_df = pd.DataFrame(columns=['f1_predictions', 'f2_predictions'],
    #                                                index=self.data.get_whole_data(return_test_and_val_only=True).index)
    #     self.predicitons_history_df['f1_predictions'], self.predicitons_history_df[
    #         'f2_predictions'] = model1_prediction, model2_prediction