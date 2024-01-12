from sklearn.ensemble import RandomForestClassifier

from model_wrapper import ModelWrapper
from reconcile import Reconcile
from fairlearn.datasets import fetch_adult
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
if __name__ == "__main__":
    data = fetch_adult(as_frame=True).frame
    data = data.drop('fnlwgt', axis=1)
    features = [
        'age',
        'education-num',
        'hours-per-week',
    ]
    target_column = "class"
    # train_data = data[features].values.astype(float)
    data[features] = data[features].astype(float)
    data[target_column] = (data[target_column] == '>50K').astype(int)
    train, test = train_test_split(data, random_state=104, test_size=0.25, shuffle=True)
    model1 = ModelWrapper(LogisticRegression(),train[features], train[target_column])
    model2 = ModelWrapper(RandomForestClassifier(),train[features], train[target_column])
    reconcile_instance = Reconcile(model1, model2, test[features+[target_column]], target_column, alpha=0.08, epsilon=0.2)
    reconcile_instance.reconcile()