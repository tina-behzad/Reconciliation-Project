from sklearn.ensemble import RandomForestClassifier
import configparser
from sklearn.linear_model import LogisticRegression

from pipeline import Pipeline

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    pipeline = Pipeline('Adult_Data', config, 'same model', LogisticRegression())
    pipeline.run()


