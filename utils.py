import ast
import datetime
import os

import pandas as pd

from wrappers.data_wrapper import Data_Wrapper


def calculate_probability_mass(complete_dataset, group):
    return len(group)/len(complete_dataset)


def round_to_fraction(v, m):
    """
    Finds the closest fraction of the form i/m to the given value v.

    Parameters:
    v (float): The value to round.
    m (int): The denominator of the fractions to consider.

    Returns:
    float: The closest value of the form i/m to v.
    """
    fractions = [i/m for i in range(-m,m + 1)]
    closest_fraction = min(fractions, key=lambda x: abs(x - v))
    return closest_fraction


def create_log_file_name(alpha, epsilon):
    current_datetime = datetime.datetime.now()
    date_str = current_datetime.strftime("%Y-%m-%d")
    hour_str = current_datetime.strftime("%H")
    return f"reconcile_log_{date_str}_{hour_str}_epsilon_{epsilon}_alpha_{alpha}"


def create_data_wrapper(data_address, target_name, categorical_features):
    data_path = os.getcwd() + data_address
    data = pd.read_csv(data_path)
    return Data_Wrapper(data, target_name, ast.literal_eval(categorical_features))
