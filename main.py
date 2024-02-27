from sklearn.ensemble import RandomForestClassifier
import configparser
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from pipeline import Pipeline

#possible models
#classifiers: (changes need to be made to modelwrapper for them to work)
# LogisticRegression
# RandomForestClassifier
#regressions
# 'Tree' for DecisionTreeRegressor
#'LinearRegression' for linear regression
#possible approaches:
#same model classifier is a string
#different features classifier is a string
#same data classifier is a list
if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('configs.ini')

    pipeline = Pipeline('Compas_Data', config, 'same model', 'LogisticRegression')
    pipeline.run()


# KNeighborsClassifier
# DecisionTreeClassifier
# LogisticRegression
# LinearRegression
# DecisionTreeRegressor
# KNeighborsRegressor