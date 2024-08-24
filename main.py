import configparser
from pipelines.experiment1_pipeline import ExperimentOnePipeline

#possible models
# LogisticRegression
# RandomForestClassifier
#DecisionTreeClassifier
#regressions
# DecisionTreeRegressor
# LinearRegression
#KNeighborsRegressor
#possible approaches:
#'same model' classifier is a string, different subsets of data will be used
#'different features' classifier is a string
#'same data' classifier is a list, different models will be used
if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('configs.ini')

    pipeline = ExperimentOnePipeline('Community_Data', config, 'same model', 'LinearRegression')
    pipeline.run()

