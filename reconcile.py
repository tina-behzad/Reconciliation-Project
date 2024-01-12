import itertools

import pandas as pd
import math
from enum import Enum
from utils import calculate_probability_mass, round_to_fraction, create_log_file_name
import logging
from model_wrapper import ModelWrapper


class Subscript(Enum):
    greater = 0
    smaller = 1


class Reconcile:
    def __init__(self, f1: ModelWrapper, f2: ModelWrapper, dataset: pd.DataFrame, target_feature_name: str, alpha: float, epsilon: float):
        """
        Initializes the Reconcile class with two models, a dataset, and parameters alpha and epsilon.

        Parameters:
        f1: The first model(it should be an instance of the ModelWrapper class).
        f2: The second model(it should be an instance of the ModelWrapper class).
        dataset: The dataset to be used.
        target_feature_name : Name of the target feature in the dataset.
        alpha: Approximate group conditional mean consistency parameter.
        epsilon: disagreement threshhold.
        """
        self.model1 = f1
        self.model2 = f2
        self.dataset = dataset
        self.dataset.insert(0, 'assigned_id', range(0, len(self.dataset)))
        self.target_feature = target_feature_name
        self.alpha = alpha
        self.epsilon = epsilon
        self.get_model_predictions()
        self.predicitons_history_df = self.dataset[['f1_predictions', 'f2_predictions']].copy()
        logging.basicConfig(filename='./logs/'+ create_log_file_name(self.alpha, self.epsilon) + ".log", encoding='utf-8',
                            level=logging.DEBUG, format='%(asctime)s %(message)s')

    def get_model_predictions(self):
        """
        Use models f1 and f2 to make predictions on the dataset.
        Add these predictions as new columns to the dataset.
        """
        feature_list = self.dataset.columns.tolist()
        feature_list = [feature_name for feature_name in feature_list if feature_name not in ["assigned_id", self.target_feature]]

        self.dataset['f1_predictions'] = self.model1.predict(self.dataset[feature_list])
        self.dataset['f2_predictions'] = self.model2.predict(self.dataset[feature_list])

    def find_disagreement_set(self):
        """
                Find sets of data points where the predictions of f1 and f2 differ significantly.
        """
        if 'f1_predictions' not in self.dataset.columns or 'f2_predictions' not in self.dataset.columns:
            self.get_model_predictions()

        diff = (self.dataset['f1_predictions'] - self.dataset['f2_predictions']).abs()
        u_epsilon = self.dataset[diff > self.epsilon]
        u_epsilon_greater = u_epsilon[self.dataset['f1_predictions'] > self.dataset['f2_predictions']]
        u_epsilon_smaller = u_epsilon[self.dataset['f1_predictions'] < self.dataset['f2_predictions']]
        return u_epsilon, u_epsilon_greater, u_epsilon_smaller

    def calculate_consistency_violation(self, u, v_star, v):
        return calculate_probability_mass(self.dataset, u) * v_star * v

    def find_candidate_for_update(self, u_greater, u_smaller):
        u = [u_greater, u_smaller]
        v_star = [u_greater[self.target_feature].mean(), u_smaller[self.target_feature].mean()]
        v = [[u_greater['f1_predictions'].mean(), u_smaller['f1_predictions'].mean()],
             [u_greater['f2_predictions'].mean(), u_smaller['f2_predictions'].mean()]]
        consistency_violation = -math.inf
        selected_subscript = -1
        selected_i = -1
        for subscript, i in itertools.product([Subscript.greater.value, Subscript.smaller.value], [0, 1]):

            new_consistency_violation = self.calculate_consistency_violation(u[subscript], v_star[subscript],
                                                                             v[i][subscript])
            # TODO: Breaking the tie is not currently arbitrary as the algorithm suggests.
            if new_consistency_violation > consistency_violation:
                consistency_violation = new_consistency_violation
                selected_subscript = subscript
                selected_i = i
        return selected_subscript, selected_i

    def patch(self, selected_model, g, delta):
        subset_ids = set(g['assigned_id'])
        self.dataset.loc[self.dataset['assigned_id'].isin(subset_ids), selected_model] += delta

    def log_round_info(self,t,u,chosen_subscript,index,delta):
        message = (f'round {t} complete. Miu(u) = {round(calculate_probability_mass(self.dataset, u),4)}.'
                   f' u {Subscript(chosen_subscript).name} f_{index+1} was updated. update value(delta) = {delta}')
        logging.info(message)

    def reconcile(self):
        t = 0
        m = round(2/(math.sqrt(self.alpha)*self.epsilon))
        u, u_greater, u_smaller = self.find_disagreement_set()
        while calculate_probability_mass(self.dataset, u) >= self.alpha:
            subscript, i = self.find_candidate_for_update(u_greater, u_smaller)
            # selected_model = self.model1 if i==0 else self.model2
            selected_model_predictions = "f1_predictions" if i==0 else "f2_predictions"
            g = u_greater if subscript == Subscript.greater.value else u_smaller
            delta = g[self.target_feature].mean() - g[selected_model_predictions].mean()
            delta = round_to_fraction(abs(delta), m) if delta > 0 else -round_to_fraction(abs(delta), m)
            self.patch(selected_model_predictions, g, delta)
            self.log_round_info(t,u,subscript, i,delta)
            self.predicitons_history_df[f'{t}_{selected_model_predictions}'] = self.dataset[selected_model_predictions].copy()
            t += 1
            u, u_greater, u_smaller = self.find_disagreement_set()
        self.predicitons_history_df.to_csv('./logs/datasets/'+ create_log_file_name(self.alpha, self.epsilon) + ".csv", index=False)


