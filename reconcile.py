import itertools
import random
import pandas as pd
import math
from enum import Enum
from utils import calculate_probability_mass, round_to_fraction, create_log_file_name
import logging
from model_wrapper import ModelWrapper
from datetime import datetime
import matplotlib.pyplot as plt


class Subscript(Enum):
    greater = 0
    smaller = 1


class Reconcile:
    def __init__(self, f1: ModelWrapper, f2: ModelWrapper, dataset: pd.DataFrame, target_feature_name: str, alpha: float, epsilon: float, trained_on_different_features = False, model_feature_lists = None):
        """
        Initializes the Reconcile class with two models, a dataset, and parameters alpha and epsilon.

        Parameters:
        f1: The first model (it should be an instance of the ModelWrapper class).
        f2: The second model (it should be an instance of the ModelWrapper class).
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
        self.trained_on_different_features = trained_on_different_features
        self.model_feature_lists = model_feature_lists
        self.get_model_predictions()
        self.predicitons_history_df = self.dataset[['f1_predictions', 'f2_predictions']].copy()
        logging.basicConfig(filename='./logs/'+ create_log_file_name(self.alpha, self.epsilon) + ".log", encoding='utf-8',
                            level=logging.DEBUG, format='%(asctime)s %(message)s')
        logging.getLogger('matplotlib.font_manager').disabled = True
        logging.info(f'model 1 name: {type(self.model1.model[1]).__name__} model 2 name: {type(self.model2.model[1]).__name__}')
        #logging.info(f'dataset name: {dataset_name}')

    def get_model_predictions(self):
        """
        Use models f1 and f2 to make predictions on the dataset.
        Add these predictions as new columns to the dataset.
        """
        if self.trained_on_different_features:
            # feature_list = self.dataset.columns.tolist()
            # exclusion_list  = ["assigned_id", self.target_feature] + self.model1_feature_list
            # model2_feature_list = [feature_name for feature_name in feature_list if
            #                 feature_name not in exclusion_list]

            self.dataset['f1_predictions'] = self.model1.predict(self.dataset[self.model_feature_lists[0]])
            self.dataset['f2_predictions'] = self.model2.predict(self.dataset[self.model_feature_lists[1]])
        else:
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
        return calculate_probability_mass(self.dataset, u) * pow((v_star - v),2)

    def find_candidate_for_update(self, u_greater, u_smaller):
        u = [u_greater, u_smaller]
        v_star = [u_greater[self.target_feature].mean(), u_smaller[self.target_feature].mean()]
        v = [[u_greater['f1_predictions'].mean(), u_smaller['f1_predictions'].mean()],
             [u_greater['f2_predictions'].mean(), u_smaller['f2_predictions'].mean()]]
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
                selected_candidates.append([subscript,i])

        # Break the tie arbitrary
        selected_candidate = random.choice(selected_candidates)
        selected_subscript = selected_candidate[0]
        selected_i = selected_candidate[1]
        return selected_subscript, selected_i

    def patch(self, selected_model, g, delta):
        subset_ids = set(g['assigned_id'])
        self.dataset.loc[self.dataset['assigned_id'].isin(subset_ids), selected_model] += delta
        #apply project
        self.dataset[selected_model] = self.dataset[selected_model].clip(lower=0, upper=1)

    def log_round_info(self,t,u,chosen_subscript,index,delta):
        message = (f'round {t} complete. Miu(u) = {round(calculate_probability_mass(self.dataset, u),4)}.'
                   f' u {Subscript(chosen_subscript).name} f_{index+1} was updated. update value(delta) = {delta}')
        logging.info(message)

    def final_round_logs(self ,brier_scores,u,t,t1,t2, time):
        rounded_brier_scores = [ [round(score[0], 3), round(score[1], 3)]  for score in brier_scores ]
        message = (f'total rounds {t} completed in {time}. T1 = {t1} T1={t2} \n'
                   f'brier_scores = {rounded_brier_scores} \n')
        print(message)
        logging.info(message)
        logging.info(self.calculate_theorem31(t,t1,t2,rounded_brier_scores[0], rounded_brier_scores[-1],u))

    def calculate_theorem31(self, t,t1,t2,initial_brier_scores,final_brier_scores, u):
        multiplier = (16/(self.alpha*pow(self.epsilon,2)))
        rounds_upper_limit = round((final_brier_scores[0]+ final_brier_scores[1])*multiplier, 3)
        brier_score_update_f1 = round(initial_brier_scores[0] - t1*(1/multiplier), 3)
        brier_score_update_f2 = round(initial_brier_scores[1] - t2*(1/multiplier), 3)
        final_mu = round(calculate_probability_mass(self.dataset, u),3)
        message = f'1. {t} <= {rounds_upper_limit} \n2. {final_brier_scores[0]} <= {brier_score_update_f1} and {final_brier_scores[1]} <= {brier_score_update_f2} \n3. {final_mu} < {self.alpha}'
        print(message)
        return message
    def reconcile(self):
        start_time = datetime.now()
        t = 0
        t1 = 0
        t2 = 0
        m = round(2/(math.sqrt(self.alpha)*self.epsilon))
        brier_scores = [[self.model1.get_brier_score(self.dataset['f1_predictions'],self.dataset[self.target_feature], True),
                                 self.model2.get_brier_score(self.dataset['f2_predictions'],self.dataset[self.target_feature], True)]]
        u, u_greater, u_smaller = self.find_disagreement_set()
        print("initial disagreement level = {}".format(calculate_probability_mass(self.dataset, u)))
        logging.info("initial disagreement level = {}".format(calculate_probability_mass(self.dataset, u)))
        while calculate_probability_mass(self.dataset, u) >= self.alpha:
            subscript, i = self.find_candidate_for_update(u_greater, u_smaller)
            # selected_model = self.model1 if i==0 else self.model2
            selected_model_predictions = "f1_predictions" if i==0 else "f2_predictions"
            g = u_greater if subscript == Subscript.greater.value else u_smaller
            delta = g[self.target_feature].mean() - g[selected_model_predictions].mean()
            delta = round_to_fraction(delta, m)
            self.patch(selected_model_predictions, g, delta)
            self.log_round_info(t,u,subscript, i,delta)
            self.predicitons_history_df[f'{t}_{selected_model_predictions}'] = self.dataset[selected_model_predictions].copy()
            brier_scores.append([self.model1.get_brier_score(self.dataset['f1_predictions'],self.dataset[self.target_feature], True),
                                 self.model2.get_brier_score(self.dataset['f2_predictions'],self.dataset[self.target_feature], True)])
            if i == 0:
                t1 += 1
            else:
                t2 += 1
            t += 1
            u, u_greater, u_smaller = self.find_disagreement_set()

        end_time = datetime.now()
        self.final_round_logs(brier_scores,u,t,t1,t2,(end_time-start_time).seconds)
        self.predicitons_history_df.to_csv('./logs/datasets/'+ create_log_file_name(self.alpha, self.epsilon) + ".csv", index=False)
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


