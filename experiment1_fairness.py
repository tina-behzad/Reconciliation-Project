import ast
import configparser
import copy
import itertools
import traceback
import warnings
from datetime import datetime
from tqdm import tqdm
import pandas as pd

from models.model_sets import classification_models, regression_models
from pipelines.fairness_pipeline import Fairness_Pipeline
from utils import create_data_wrapper

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    config = configparser.ConfigParser()
    config.read('configs.ini')
    experiment_1_results_csv_file_name = "./results/experiment1_fairness_" + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    exp_1_columns = ["Data", "Method", "MSE", 'ratio']
    results_1_df = pd.DataFrame(
        columns=exp_1_columns).to_csv(experiment_1_results_csv_file_name, index=False)
    alpha = float(config['Reconciliation_Configs']['Alpha'])
    epsilon = float(config['Reconciliation_Configs']['Epsilon'])
    # model_similarity_threshold = float(config['Training_Configs']['Model_Similarity_Threshhold'])
    model_similarity_threshold = 0.02
    Experiment_Repeat_Numbers = int(config['Training_Configs']['Experiment_Repeat_Numbers'])
    dataset_set_size = {}
    for i in tqdm(range(Experiment_Repeat_Numbers)):
        for dataset_name in ast.literal_eval(config['Training_Configs']['Datasets']):
            data = create_data_wrapper(config[dataset_name]['Address'], config[dataset_name]['Target_Col'],
                                       config[dataset_name]['Categorical_Features'], sensitive_features= ast.literal_eval(config[dataset_name]['sensitive_features']))
            is_classification = ast.literal_eval(config[dataset_name]['Classification'])
            models = classification_models if is_classification else regression_models
            models_pipeline = [copy.deepcopy(model) for model in models]
            result_1_dict = {"Data": dataset_name}
            pipeline = Fairness_Pipeline(data, models_pipeline, is_classification, alpha, epsilon,
                                                model_similarity_threshold)
            pipeline.run(experiment_1_results_csv_file_name, result_1_dict,set_or_pair='pair')
