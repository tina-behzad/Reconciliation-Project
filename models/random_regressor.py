import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
class RandomRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, X, y):
        # Determine lower and upper bounds from the target variable
        self.lower_bound = np.min(y)
        self.upper_bound = np.max(y)
        return self

    def predict(self, X):
        if self.lower_bound is None or self.upper_bound is None:
            raise ValueError(
                "This RandomRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        # Generate random predictions within the determined bounds
        return np.random.uniform(self.lower_bound, self.upper_bound, X.shape[0])