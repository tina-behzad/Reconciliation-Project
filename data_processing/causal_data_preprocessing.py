import pandas as pd
import numpy as np

from IPython.display import Image

from causalml.dataset import make_uplift_classification
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot
from sklearn.model_selection import train_test_split



def get_leaf_node(decisionTree, row, x_names):
    """
    Traverse the tree to find the leaf node for a given row.

    Args
    ----
    decisionTree : object
        The trained decision tree object.

    row : pd.Series
        A single row from the DataFrame.

    x_names : list
        List of feature names.

    Returns
    -------
    The index of the leaf node where the row falls.
    """
    while decisionTree.results is None:  # Traverse until a leaf node
        # Get the feature name and value to compare
        col_name = x_names[decisionTree.col]
        value = row[col_name]

        # Traverse the true or false branch based on the comparison
        if value >= decisionTree.value:
            decisionTree = decisionTree.trueBranch
        else:
            decisionTree = decisionTree.falseBranch

    # Return the unique leaf node identifier (could be a hash or index)
    return id(decisionTree)

def preprocess_twin():
    x = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_X_3years_samesex.csv")

    # The outcome data contains mortality of the lighter and heavier twin
    y = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv")

    # The treatment data contains weight in grams of both the twins
    t = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_T_3years_samesex.csv")
    lighter_columns = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
                       'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
                       'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
                       'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
                       'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
                       'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
                       'data_year', 'nprevistq', 'dfageq', 'feduc6', 'infant_id_0',
                       'dlivord_min', 'dtotord_min', 'bord_0',
                       'brstate_reg', 'stoccfipb_reg', 'mplbir_reg']
    heavier_columns = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
                       'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
                       'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
                       'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
                       'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
                       'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
                       'data_year', 'nprevistq', 'dfageq', 'feduc6',
                       'infant_id_1', 'dlivord_min', 'dtotord_min', 'bord_1',
                       'brstate_reg', 'stoccfipb_reg', 'mplbir_reg']

    data = []

    for i in range(len(t.values)):

        # select only if both <=2kg
        if t.iloc[i].values[1] >= 2000 or t.iloc[i].values[2] >= 2000:
            continue

        this_instance_lighter = list(x.iloc[i][lighter_columns].values)
        this_instance_heavier = list(x.iloc[i][heavier_columns].values)

        # adding weight
        this_instance_lighter.append(t.iloc[i].values[1])
        this_instance_heavier.append(t.iloc[i].values[2])

        # adding treatment, is_heavier
        this_instance_lighter.append(0)
        this_instance_heavier.append(1)

        # adding the outcome
        this_instance_lighter.append(y.iloc[i].values[1])
        this_instance_heavier.append(y.iloc[i].values[2])
        data.append(this_instance_lighter)
        data.append(this_instance_heavier)

        cols = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
                'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
                'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
                'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
                'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
                'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
                'data_year', 'nprevistq', 'dfageq', 'feduc6',
                'infant_id', 'dlivord_min', 'dtotord_min', 'bord',
                'brstate_reg', 'stoccfipb_reg', 'mplbir_reg', 'wt', 'treatment', 'outcome']
    df = pd.DataFrame(columns=cols, data=data)
    df = df.astype({"treatment": 'bool'}, copy=False)  # explicitly assigning treatment column as boolean

    df.fillna(value=df.mean(), inplace=True)  # filling the missing values
    df.fillna(value=df.mode().loc[0], inplace=True)
    cols.remove('outcome')
    cols.remove('treatment')

    df['treatment'] = np.where(df['treatment'] == 0, 'control', 'treatment1')
    ctree = UpliftTreeClassifier(control_name='control', max_depth=15, min_samples_leaf=40)
    ctree.fit(X=df[cols].values,
              treatment=df['treatment'].values.flatten(),
              y=df['outcome'].values.flatten())
    df['leaf_node'] =  df.apply(lambda row: get_leaf_node(ctree.fitted_uplift_tree, row, cols), axis=1)
    unique_leaf_nodes = df['leaf_node'].unique()
    leaf_node_mapping = {original_value: new_value for new_value, original_value in
                         enumerate(unique_leaf_nodes, start=1)}
    df['leaf_node'] = df['leaf_node'].map(leaf_node_mapping)
    df.to_csv('../data/causal_data/twin_data_grouped.csv', index=False)




if __name__ == '__main__':
    preprocess_twin()