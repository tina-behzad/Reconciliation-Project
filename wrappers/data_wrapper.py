
import pandas as pd
from sklearn.model_selection import train_test_split


class Data_Wrapper():
    def __init__(self,data,target_feature_name, categorical_features, split_ratio=0.2):
        x_dummies = pd.get_dummies(data.drop(columns=[target_feature_name],axis=1),columns=categorical_features, drop_first=True)
        X_train, self.test_x, y_train, self.test_y = train_test_split(x_dummies, data[target_feature_name], test_size=split_ratio)
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(X_train, y_train, test_size=(split_ratio/(1-split_ratio))) #0.25 x 0.8 = 0.2
        # self.train_X = train_X
        # self.train_y = train_y
        # self.test_x = test_X
        # self.test_y = test_y
        self.target_feature_name = target_feature_name
        # self.train_predicted = None
        # self.test_predicted = None

    def get_whole_data(self, return_test_and_val_only = False):
        if return_test_and_val_only:
            return pd.concat([self.val_X, self.test_x])
        return pd.concat([self.train_X,self.val_X, self.test_x])

    # def set_prediction(self, complete_predictions):
    #     self.train_predicted = complete_predictions[:self.train_X.shape[0]]
    #     self.test_predicted = complete_predictions[self.train_X.shape[0]:]

    # def remove_prediction(self):
    #     self.train_predicted = None
    #     self.test_predicted = None

    def get_all_labels(self, return_test_and_val_only = False):
        if return_test_and_val_only:
            return pd.concat([self.val_y,self.test_y])
        return pd.concat([self.train_y,self.val_y,self.test_y])

    # def get_test_prediction(self, complete_predictions):
    #     return complete_predictions[self.train_X.shape[0]:]
    #
    # def get_train_prediction(self, complete_predictions):
    #     return complete_predictions[self.train_X.shape[0]:]

    def seperate_data_section(self,given_dataset, return_section="train"):
        if return_section == "train":
            return given_dataset.loc[given_dataset.index.intersection(self.train_X.index)]
        elif return_section == "test":
            return given_dataset.loc[given_dataset.index.intersection(self.test_x.index)]
        elif return_section == "val":
            return given_dataset.loc[given_dataset.index.intersection(self.val_X.index)]

    def seperate_intersection_data(self, seperation_data,intersection_data, return_section="train"):
        if return_section == "train":
            common_indices = intersection_data.index.intersection(self.train_X.index)
        elif return_section == "test":
            common_indices = intersection_data.index.intersection(self.test_x.index)
        elif return_section == "val":
            common_indices = intersection_data.index.intersection(self.val_X.index)
        elif return_section == "test_val":
            common_indices = intersection_data.index.intersection(self.val_X.index).union(intersection_data.index.intersection(self.test_x.index))
        return seperation_data.loc[common_indices]

    def return_intersection_true_label(self,given_dataset, return_section):
        if return_section=="train":
            return self.train_y.loc[given_dataset.index.intersection(self.train_y.index)]
        elif return_section=="test":
            return self.test_y.loc[given_dataset.index.intersection(self.test_y.index)]
        elif return_section=="val":
            return self.val_y.loc[given_dataset.index.intersection(self.val_y.index)]


