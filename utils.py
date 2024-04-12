import datetime
import random



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




# def apply_reconcile(model1, model2, X_test, target_col_name,alpha,epsilon):
#     reconcile_instance = Reconcile(model1, model2, X_test.copy(), target_col_name, alpha,
#                                    epsilon, False, [])
#     u, _, __ = reconcile_instance.find_disagreement_set()
#     current_models_disagreement_set_probability_mass = calculate_probability_mass(X_test, u)
#     if current_models_disagreement_set_probability_mass > alpha:
#         scores = reconcile_instance.reconcile()
#         model1_predictions, model2_predictions = reconcile_instance.get_reconciled_predictions()
#         model1.set_reconcile(model1_predictions)
#         model2.set_reconcile(model2_predictions)
#     return random.choice([model1,model2])