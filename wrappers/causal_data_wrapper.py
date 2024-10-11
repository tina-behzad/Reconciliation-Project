from wrappers.data_wrapper import Data_Wrapper
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


class CausalDataWrapper(Data_Wrapper):
    def __init__(self, data,outcome_column,group_column,split_ratio=0.5):
        self.test_x, self.val_X, self.test_y, self.val_y = train_test_split(data.drop(columns = [outcome_column]), data[outcome_column], test_size=split_ratio)
        self.group_column = group_column
        self.target_feature_name = outcome_column
