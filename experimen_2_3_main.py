import ast
import configparser
import itertools
import traceback
from datetime import datetime

import pandas as pd

from models.model_sets import classification_models, regression_models
from pipelines.experiment1_pipeline import ExperimentOnePipeline
from pipelines.experiment2_pipeline import ExperimentTwoPipeline
from utils import create_data_wrapper

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('configs.ini')
    experiment_2_results_csv_file_name = "./results/experiment2 " + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    experiment_3_results_csv_file_name = "./results/experiment3 " + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    exp_2_columns = ["Data","Method","MSE"]
    exp_3_columns = ["Data","Method","Measure"]
    results_2_df = pd.DataFrame(
        columns=exp_2_columns).to_csv(experiment_2_results_csv_file_name, index=False)
    results_3_df = pd.DataFrame(
        columns=exp_3_columns).to_csv(experiment_3_results_csv_file_name, index=False)
    alpha = float(config['Reconciliation_Configs']['Alpha'])
    epsilon = float(config['Reconciliation_Configs']['Epsilon'])
    model_similarity_threshold = float(config['Training_Configs']['Model_Similarity_Threshhold'])
    for dataset_name in ast.literal_eval(config['Training_Configs']['Datasets']):
        data = create_data_wrapper(config[dataset_name]['Address'], config[dataset_name]['Target_Col'],
                                   config[dataset_name]['Categorical_Features'])
        is_classification = ast.literal_eval(config[dataset_name]['Classification'])
        models = classification_models if is_classification else regression_models
        result_2_dict = {"Data": dataset_name}
        exp_2_pipeline = ExperimentTwoPipeline(data,models,is_classification,alpha,epsilon,model_similarity_threshold)
        exp_2_pipeline.run(experiment_2_results_csv_file_name,result_2_dict)

        # for method in ast.literal_eval(config['Training_Configs']['Possible_approaches']):
        #     if method == 'Dummy':
        #         pipeline = ExperimentOnePipeline(data, method, 'Dummy', is_classification, alpha, epsilon,
        #                                          model_similarity_threshold)
        #         result_dict = {"Data": dataset_name, "Method": method, "Models": 'Dummy'}
        #         pipeline.run(results_csv_file_name, result_dict)
        #     elif method == 'different models':
        #         for model1, model2 in itertools.combinations(models, 2):
        #             try:
        #                 pipeline = ExperimentOnePipeline(data, method, [model1, model2], is_classification, alpha, epsilon,
        #                                                  model_similarity_threshold)
        #                 result_dict = {"Data": dataset_name.replace("_Data",""),"Method":method, "Models":[model1, model2]}
        #                 pipeline.run(results_csv_file_name, result_dict)
        #             except:
        #                 print(traceback.format_exc())
        #     else:
        #         for model in models:
        #             try:
        #                 pipeline = ExperimentOnePipeline(data, method, model, is_classification, alpha, epsilon,
        #                                                  model_similarity_threshold)
        #                 result_dict = {"Data": dataset_name,"Method": method, "Models": model}
        #                 pipeline.run(results_csv_file_name, result_dict)
        #             except:
        #                 print(traceback.format_exc())
