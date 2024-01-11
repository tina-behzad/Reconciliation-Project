
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
