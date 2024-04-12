import ast
import logging
import math
import os
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import random

from model_wrapper import ModelWrapper
from reconcile import Reconcile
from utils import create_log_file_name, calculate_probability_mass


class Experiment:
    def __init__(self,dataset_name, config, models):
        """
        Initializes the Experiment class with a dataset and a set of models.

        Parameters:
        - dataset: The dataset to be used in the experiment.
        - models: A set of models (M) to be evaluated.
        """
        self.config = config
        self.target_variable_name = self.config[dataset_name]['Target_Col']
        self.sensitive_features = ast.literal_eval(self.config[dataset_name]['sensitive_features'])
        self.dataset_name = dataset_name
        self.alpha = float(config['Reconciliation_Configs']['Alpha'])
        self.epsilon = float(config['Reconciliation_Configs']['Epsilon'])
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data(dataset_name)
        logging.basicConfig(filename='./logs/' + create_log_file_name(self.alpha, self.epsilon) + ".log", encoding='utf-8',
                            level=logging.DEBUG, format='%(asctime)s %(message)s')
        logging.getLogger('matplotlib.font_manager').disabled = True
        self.set_M = self.find_set_M(models)


    def prepare_data(self, dataset_name):
        data_path = os.getcwd() + self.config[dataset_name]['Address']
        data = pd.read_csv(data_path)
        data = pd.get_dummies(data, columns=ast.literal_eval(self.config[dataset_name]['Categorical_Features']), drop_first=True)
        target_col_name = self.config[dataset_name]['Target_Col']
        X,y = data.drop(columns=[target_col_name], axis = 1), data[target_col_name]
        X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train,X_test,y_train,y_test

    def find_set_M(self,classifiers):
        performance_scores = {}
        for name, model in classifiers:
            scaled_model = Pipeline([
                ('normalizer', StandardScaler()),
                ('classifier', model)])
            scores = cross_val_score(scaled_model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            performance_scores[name] = scores.mean()
            print("{} is done".format(name))

        top_score = max(performance_scores.values())
        threshold = top_score - float(self.config['Training_Configs']['Model_Similarity_Threshhold'])

        selected_models = [(name, model) for (name, model) in classifiers if performance_scores[name] >= threshold]
        logging.info('Selected models that perform within 5% of the top model accuracy:')
        for (name, _) in selected_models:
            logging.info(f"- {name}: {performance_scores[name]:.4f}")

        M = [Pipeline([
            ('normalizer', StandardScaler()),
            ('classifier', model)]) for (name, model) in selected_models]
        return M

    def run_experiment(self):
        variances = []
        set_M_predictions = self.get_set_M_predictions()
        # logging.info("#########For set M##########")
        # variances.append(self.calculate_metrics(set_M_predictions))
        # all_combinations_predictions = self.get_all_combinations_reconciled()
        # logging.info("#########For set M prime with all combinations##########")
        # self.calculate_metrics(all_combinations_predictions)
        # only_one_chosen_after_reconcile_predictions = all_combinations_predictions[0::2]
        # logging.info("#########For set M prime only choosing one model after reconcile##########")
        # self.calculate_metrics(only_one_chosen_after_reconcile_predictions)
        # randomly_selected_after_reconcile = random.sample(all_combinations_predictions, len(self.set_M))
        # logging.info("#########For set M prime choosing size(M) randomly##########")
        # self.calculate_metrics(randomly_selected_after_reconcile)
        contesting_set_predictions = self.get_contesting_set()
        logging.info("#########contesting##########")
        self.calculate_metrics(contesting_set_predictions)


    def calculate_metrics(self, prediction_lists):
        predictions = pd.DataFrame.from_dict({"model_"+ str(i) : prediction_lists[i] for i in range(0,len(prediction_lists))})
        variance_in_perdictions = predictions.var(axis=1)
        logging.info(stats.describe(variance_in_perdictions))
        ambiguity = self.calculate_ambiguity(predictions)
        logging.info(f"ambiguity for the set over predictions is {ambiguity:.4f}")
        discrepency = self.calculate_discrepancy(predictions)
        logging.info(f"discrepancy for the set over predictions is {discrepency:.4f}")
        logging.info("Disagreement values")
        logging.info(stats.describe(self.calculate_disagreement(predictions)))
        return variance_in_perdictions


    def get_all_combinations_reconciled(self):
        self.X_test[self.target_variable_name] = self.y_test
        model_predictions = []
        for model1, model2 in combinations(self.set_M, 2):
            model_wrapper1 = ModelWrapper(model1, self.X_train, self.y_train)
            model_wrapper2 = ModelWrapper(model2, self.X_train, self.y_train)
            reconcile_instance = Reconcile(model_wrapper1, model_wrapper2, self.X_test.copy(), self.target_variable_name, self.alpha,
                                           self.epsilon, False, [])
            u, _, __ = reconcile_instance.find_disagreement_set()
            current_models_disagreement_set_probability_mass = calculate_probability_mass(self.X_test, u)

            if current_models_disagreement_set_probability_mass > self.alpha:
                scores = reconcile_instance.reconcile()
                model1_predictions, model2_predictions = reconcile_instance.get_reconciled_predictions()
                model_predictions.append(model1_predictions)
                model_predictions.append(model2_predictions)
                # model1.set_reconcile(model1_predictions)
                # model2.set_reconcile(model2_predictions)
            else:
                print("not reconciled")
        return model_predictions

    def get_contesting_set(self):
        model_predictions = []
        models = self.set_M.copy()
        model1 = random.choice(models)
        models.remove(model1)
        model2 = random.choice(models)
        models.remove(model2)
        model_wrapper1 = ModelWrapper(model1, self.X_train, self.y_train)
        model_wrapper2 = ModelWrapper(model2, self.X_train, self.y_train)
        self.X_test[self.target_variable_name] = self.y_test
        reconcile_instance = Reconcile(model_wrapper1, model_wrapper2, self.X_test.copy(), self.target_variable_name, self.alpha,
                                       self.epsilon, False, [])
        u, _, __ = reconcile_instance.find_disagreement_set()
        current_models_disagreement_set_probability_mass = calculate_probability_mass(self.X_test, u)
        if current_models_disagreement_set_probability_mass > self.alpha:
            scores = reconcile_instance.reconcile()
            model1_predictions, model2_predictions = reconcile_instance.get_reconciled_predictions()
            model_wrapper1.set_reconcile(model1_predictions)
            model_wrapper2.set_reconcile(model2_predictions)
        chosen_model =  random.choice([model_wrapper1, model_wrapper2])
        if not chosen_model.reconciled:
            chosen_model.set_reconcile(chosen_model.predict(self.X_test.drop(self.target_variable_name, axis = 1)))
        model_predictions.append(chosen_model.predict(None))
        while models:
            next_model = random.choice(models)
            models.remove(next_model)
            model_wrapper = ModelWrapper(next_model, self.X_train, self.y_train)
            reconcile_instance = Reconcile(chosen_model, model_wrapper, self.X_test.copy(), self.target_variable_name, self.alpha,
                                           self.epsilon, False, [])
            u, _, __ = reconcile_instance.find_disagreement_set()
            current_models_disagreement_set_probability_mass = calculate_probability_mass(self.X_test, u)
            if current_models_disagreement_set_probability_mass > self.alpha:
                scores = reconcile_instance.reconcile()
                model1_predictions, model2_predictions = reconcile_instance.get_reconciled_predictions()
                chosen_model.set_reconcile(model1_predictions)
                model_wrapper.set_reconcile(model2_predictions)
            chosen_model = random.choice([model_wrapper, chosen_model])
            if not chosen_model.reconciled:
                chosen_model.set_reconcile(chosen_model.predict(self.X_test.drop(self.target_variable_name, axis=1)))
            model_predictions.append(chosen_model.predict(None))
        return model_predictions

    def get_set_M_predictions(self):
        predictions = []
        for model in self.set_M:
            model.fit(self.X_train, self.y_train)
            predictions.append(model.predict_proba(self.X_test)[:, 1])
        return predictions

    def calculate_ambiguity(self, predictions):
        max_abs_diff_per_point = predictions.apply(lambda x: x.max() - x.min(), axis=1)
        total_diff = max_abs_diff_per_point.sum()/len(max_abs_diff_per_point)
        return total_diff

    def calculate_discrepancy(self, predictions):
        max_diff = -math.inf
        # Generate all combinations of column pairs
        for col1, col2 in combinations(predictions.columns, 2):
            diff_summ = np.sum(abs(predictions[col1] - predictions[col2]))/predictions.shape[0]
            if diff_summ > max_diff:
                max_diff = diff_summ

        return max_diff

    def calculate_disagreement(self, predictions):
        disagreements = []
        # Generate all combinations of column pairs
        for col1, col2 in combinations(predictions.columns, 2):
            diff = (predictions[col1] - predictions[col2]).abs()
            u_epsilon = predictions[diff > self.epsilon]
            disagreements.append(calculate_probability_mass(self.X_test,u_epsilon))
        return disagreements
