from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
class ModelWrapper:
    def __init__(self, model, train_x, train_y, ):
        """
        Initializes the ModelWrapper with a scikit-learn model.

        Parameters:
        model (estimator): An instance of a scikit-learn model.
        """
        self.model = Pipeline([
        ('normalizer', StandardScaler()),
        ('classifier', model),])
        self.model.fit(train_x, train_y)


    def predict(self, X, return_probabilities=True):
        """
        Predicts outcomes for samples in X.

        Parameters:
        X (array-like): Input data for making predictions.
        return_probabilities (bool, optional): If True, returns class probabilities.
                                                Otherwise, returns class predictions.
                                                Defaults to True.

        Returns:
        array-like: Predicted class probabilities or class predictions.
        """
        if not hasattr(self.model, 'classes_'):
            raise ValueError("Model has not been fitted yet. Please fit the model before calling predict.")
        if return_probabilities:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)[:, 1]
            else:
                raise AttributeError("This model does not support probability predictions.")
        else:
            return self.model.predict(X)


    def get_brier_score(self, data, labels):
        predictions = self.predict(data)
        return brier_score_loss(labels, predictions)

    def save_model_to_file(self, model_name):
        pass
        # model_name = type(model).__name__
        # with open(model_name + '.pkl', 'wb') as fid:
        #     cPickle.dump(gnb, fid)
