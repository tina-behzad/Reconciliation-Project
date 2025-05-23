import ast
import configparser
import copy
import warnings
from datetime import datetime
from tqdm import tqdm
import pandas as pd

from models.model_sets import classification_models, regression_models
from pipelines.stability_pipeline import Stability_Pipeline
from utils import create_data_wrapper

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    config = configparser.ConfigParser()
    config.read('configs.ini')
    experiment_2_results_csv_file_name = "./results/experiment2_stability_" + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    exp_2_columns = ["Data","Set Size","Individual MSE","number of random models","Method","MSE"]
    results_2_df = pd.DataFrame(
        columns=exp_2_columns).to_csv(experiment_2_results_csv_file_name, index=False)
    alpha = float(config['Reconciliation_Configs']['Alpha'])
    epsilon = float(config['Reconciliation_Configs']['Epsilon'])
    model_similarity_threshold = float(config['Training_Configs']['Model_Similarity_Threshhold'])
    Experiment_Repeat_Numbers = int(config['Training_Configs']['Experiment_Repeat_Numbers'])
    dataset_set_size = {}
    for i in tqdm(range(Experiment_Repeat_Numbers)):
        for dataset_name in ast.literal_eval(config['Training_Configs']['Datasets']):
            data = create_data_wrapper(config[dataset_name]['Address'], config[dataset_name]['Target_Col'],
                                       config[dataset_name]['Categorical_Features'])
            is_classification = ast.literal_eval(config[dataset_name]['Classification'])
            models = classification_models if is_classification else regression_models
            models_pipeline2 = [copy.deepcopy(model) for model in models]
            result_2_dict = {"Data": dataset_name}
            if i == 0:
                exp_2_pipeline = Stability_Pipeline(data, models_pipeline2, is_classification, alpha, epsilon,
                                                    model_similarity_threshold)
                set_size = exp_2_pipeline.run(experiment_2_results_csv_file_name,result_2_dict)
                dataset_set_size[dataset_name] = set_size
            else:
                exp_2_pipeline = Stability_Pipeline(data, models_pipeline2, is_classification, alpha, epsilon,
                                                    model_similarity_threshold, dataset_set_size[dataset_name])
                exp_2_pipeline.run(experiment_2_results_csv_file_name, result_2_dict)