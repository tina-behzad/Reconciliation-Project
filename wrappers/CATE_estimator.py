from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor
from causalml.inference.tree import CausalTreeRegressor, CausalRandomForestRegressor
from sklearn.ensemble._forest import ForestRegressor
from xgboost import XGBRegressor
class CATE_estimator:
    def __init__(self, estimator_name, control_column_value=0):
        self.estimator = self.create_estimator(estimator_name, control_column_value)
        self.estimator_name = estimator_name

    def create_estimator(self, estimator_name, control_column_value):
        ['R_learner', 'S_learner', 'T_learner', 'X_learner', 'Tree', 'Forest']
        if estimator_name == 'R_learner':
            return BaseSRegressor(XGBRegressor(), control_name=control_column_value)
        elif estimator_name == 'S_learner':
            return BaseSRegressor(XGBRegressor(), control_name=control_column_value)
        elif estimator_name == 'T_learner':
            return BaseTRegressor(XGBRegressor(), control_name=control_column_value)
        elif estimator_name == 'X_learner':
            return BaseXRegressor(XGBRegressor(), control_name=control_column_value)
        elif estimator_name == 'Tree':
            return CausalTreeRegressor()
        elif estimator_name == 'Forest':
            return CausalRandomForestRegressor()


    def fit(self, X,treatment, y):
        self.estimator.fit(X=X, y=y, treatment=treatment)

    def predict(self, X):
        return self.estimator.predict(X)