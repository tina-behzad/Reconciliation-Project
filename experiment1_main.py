import ast
import configparser
import itertools
import traceback
from datetime import datetime
import warnings
import pandas as pd
from tqdm import tqdm
from pipelines.experiment1_pipeline import ExperimentOnePipeline
from utils import create_data_wrapper

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    config = configparser.ConfigParser()
    config.read('configs.ini')
    results_csv_file_name = "./results/experiment1 " + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    columns = ["Data","Method", "Models", "Initial Disagreement", "Final Disagreement", "Initial Brier", "Final Brier",
               "Theorem 3.1", "T1", "T2", "Brier Scores"]
    results_df = pd.DataFrame(
        columns=columns).to_csv(results_csv_file_name, index=False)
    alpha = float(config['Reconciliation_Configs']['Alpha'])
    epsilon = float(config['Reconciliation_Configs']['Epsilon'])
    model_similarity_threshold = float(config['Training_Configs']['Model_Similarity_Threshhold'])
    Experiment_Repeat_Numbers = int(config['Training_Configs']['Experiment_Repeat_Numbers'])
    for i in tqdm(range(Experiment_Repeat_Numbers)):
        for dataset_name in ast.literal_eval(config['Training_Configs']['Datasets']):
            data = create_data_wrapper(config[dataset_name]['Address'], config[dataset_name]['Target_Col'],
                                       config[dataset_name]['Categorical_Features'])
            is_classification = ast.literal_eval(config[dataset_name]['Classification'])
            models = ast.literal_eval(config["Training_Configs"]['Classification_models']) if is_classification \
                else ast.literal_eval(config["Training_Configs"]['Regression_Models'])
            for method in ast.literal_eval(config['Training_Configs']['Possible_approaches']):
                if method == 'Dummy':
                    pipeline = ExperimentOnePipeline(data, method, 'Dummy', is_classification, alpha, epsilon,
                                                     model_similarity_threshold)
                    result_dict = {"Data": dataset_name, "Method": method, "Models": 'Dummy'}
                    pipeline.run(results_csv_file_name, result_dict)
                elif method == 'different models':
                    for model1, model2 in itertools.combinations(models, 2):
                        try:
                            pipeline = ExperimentOnePipeline(data, method, [model1, model2], is_classification, alpha, epsilon,
                                                             model_similarity_threshold)
                            result_dict = {"Data": dataset_name.replace("_Data",""),"Method":method, "Models":[model1, model2]}
                            pipeline.run(results_csv_file_name, result_dict)
                        except:
                            print(traceback.format_exc())
                else:
                    for model in models:
                        try:
                            pipeline = ExperimentOnePipeline(data, method, model, is_classification, alpha, epsilon,
                                                             model_similarity_threshold)
                            result_dict = {"Data": dataset_name,"Method": method, "Models": model}
                            pipeline.run(results_csv_file_name, result_dict)
                        except:
                            print(traceback.format_exc())
