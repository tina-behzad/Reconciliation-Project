from sklearn.base import BaseEstimator
from collections import Counter
import numpy as np
from sklearn.metrics import brier_score_loss, mean_squared_error
import random

from wrappers.data_wrapper import Data_Wrapper


class ModelAggregator(BaseEstimator):
    def __init__(self, models, technique):
        self.models = models
        self.technique = technique

    def predict(self, data: Data_Wrapper):
        if self.technique == 'mode':
            predictions =  self._predict_mode(data.test_x)
        elif self.technique == 'mean':
            predictions = self._predict_mean(data.test_x)
        elif self.technique == 'randomized':
            predictions = self._predict_randomized(data.test_x)
        elif self.technique == 'random_selection':
            predictions = self._predict_random_selection(data.test_x)
        else:
            raise ValueError("Unknown technique specified.")
        return predictions

    def _predict_mode(self, X):
        # Collect predictions from all models and take the mode
        predictions = np.array([model.predict(X) for model in self.models])
        mode_predictions = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], 0, predictions)
        return mode_predictions

    def _predict_randomized(self, X):
        # Randomly select a model for each sample and use its prediction
        randomized_predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            model = random.choice(self.models)
            randomized_predictions[i] = model.predict(X.iloc[[i]])[0]
        return randomized_predictions

    def _predict_random_selection(self, X):
        # Randomly select a model and use it for all predictions
        model = random.choice(self.models)
        return model.predict(X)

    def _predict_mean(self, X):
        # Collect predictions from all models and take the mean
        predictions = np.array([model.predict(X) for model in self.models])
        mean_predictions = np.mean(predictions, axis=0)
        return mean_predictions

