from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn import linear_model, svm, tree, ensemble, dummy,neighbors
from sklearn.base import ClassifierMixin, RegressorMixin

import models
from models.random_regressor import RandomRegressor


# TODO: move feature list the model is trained on inside wrapper
class ModelWrapper:
    def __init__(self, model, train_x, train_y,return_prob, **kwargs ):
        """
        Initializes the ModelWrapper with a scikit-learn model.

        Parameters:
        model (string): An instance of a scikit-learn model.
        """
        if isinstance(model, str):
            self.model = Pipeline([
                ('normalizer', StandardScaler()),
                ('classifier', self._get_model_instance(model, **kwargs))])
            self.model.fit(train_x, train_y)
        else:
            self.model = model

        # self.return_probs = issubclass(type(self.model), ClassifierMixin)
        self.return_probs = return_prob
    def _get_model_instance(self, model_name, **kwargs):
        # Dictionary mapping model names to their respective modules in scikit-learn
        if model_name == "Dummy":
            if self.return_probs:
                return dummy.DummyClassifier(strategy='uniform')
            else:
                return RandomRegressor()
        model_modules = {
            'LinearRegression': linear_model,
            'LogisticRegression': linear_model,
            'DecisionTreeRegressor': tree,
            'RandomForest': ensemble,
            'KNeighborsClassifier': neighbors,
            'DecisionTreeClassifier': tree,
            'KNeighborsRegressor': neighbors
            # Add more models and their respective modules as needed
        }

        if model_name not in model_modules:
            raise ValueError(f"Model {model_name} is not supported.")

        model_class = getattr(model_modules[model_name], model_name)
        return model_class(**kwargs)

    def predict(self, X ):
        """
        Predicts outcomes for samples in X.

        Parameters:
        X (array-like): Input data for making predictions.

        Returns:
        array-like: Predicted class probabilities or class predictions.
        """
        # if not hasattr(self.model, 'classes_'):
        #     raise ValueError("Model has not been fitted yet. Please fit the model before calling predict.")
        if self.return_probs:
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)


    def get_brier_score(self, data, labels, predictions = False):
        if predictions:
            return mean_squared_error(labels, data)
            # return brier_score_loss(labels, data)
        predictions = self.predict(data)
        return mean_squared_error(labels, predictions)
        # return brier_score_loss(labels, predictions)

