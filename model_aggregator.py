from sklearn.base import BaseEstimator
from collections import Counter
import numpy as np
from sklearn.metrics import brier_score_loss, mean_squared_error
import random

class ModelAggregator(BaseEstimator):
    def __init__(self, models,train_X, train_y, technique='mode'):
        self.models = models
        self.fit(train_X,train_y)
        self.technique = technique

    def fit(self, X, y):
        # Fit all models on the training data
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X, y):
        if self.technique == 'mode':
            predictions =  self._predict_mode(X)
        elif self.technique == 'randomized':
            predictions = self._predict_randomized(X)
        elif self.technique == 'random_selection':
            predictions = self._predict_random_selection(X)
        else:
            raise ValueError("Unknown technique specified.")
        return mean_squared_error(y,predictions)

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