import datetime
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
    fractions = [i/m for i in range(m + 1)]
    closest_fraction = min(fractions, key=lambda x: abs(x - v))
    return closest_fraction


def create_log_file_name(alpha, epsilon):
    current_datetime = datetime.datetime.now()
    date_str = current_datetime.strftime("%Y-%m-%d")
    hour_str = current_datetime.strftime("%H")
    return f"reconcile_log_{date_str}_{hour_str}_epsilon_{epsilon}_alpha_{alpha}"
