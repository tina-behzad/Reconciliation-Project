from datetime import datetime

import pandas as pd
import numpy as np

from IPython.display import Image
from tqdm import tqdm
from xgboost import XGBRegressor
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor
from causalml.dataset import make_uplift_classification
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot
from sklearn.model_selection import train_test_split
from causalml.inference.tree import CausalRandomForestRegressor
from reconcile import Reconcile
from wrappers.causal_data_wrapper import CausalDataWrapper
from wrappers.data_wrapper import Data_Wrapper
from wrappers.model_wrapper import ModelWrapper


def get_leaf_paths(tree_structure):
    # Number of nodes in the tree
    n_nodes = tree_structure.node_count
    # Initialize lists for left and right children
    children_left = tree_structure.children_left
    children_right = tree_structure.children_right
    # Initialize feature indices and thresholds
    feature_indices = tree_structure.feature
    thresholds = tree_structure.threshold

    # Function to trace the path of a node to the root
    def find_path(node_id, current_path):
        parent = node_parent[node_id]
        # If node is not the root, keep finding parent
        if parent != -1:
            # Determine if the node is left or right child
            if children_left[parent] == node_id:
                split = 'left'
            else:
                split = 'right'
            # Append condition to path
            condition = (feature_indices[parent], thresholds[parent], split)
            current_path.append(condition)
            # Recurse until the root is reached
            find_path(parent, current_path)

    # Identify each node's parent
    node_parent = np.full(n_nodes, -1)
    for node_id in range(n_nodes):
        if children_left[node_id] != -1:
            node_parent[children_left[node_id]] = node_id
        if children_right[node_id] != -1:
            node_parent[children_right[node_id]] = node_id

    # Dictionary to store paths for each leaf node
    leaf_paths = {}

    # Traverse nodes and find paths for leaf nodes
    for node_id in range(n_nodes):
        # Check if it's a leaf node (no children)
        if children_left[node_id] == -1 and children_right[node_id] == -1:
            path = []
            find_path(node_id, path)
            # Store the path in the dictionary, reversing it to start from the root
            leaf_paths[node_id] = path[::-1]

    return leaf_paths

if __name__ == "__main__":
    df = pd.read_csv('./data/causal_data/twin_data_grouped.csv')
    results_csv_file_name = "./results/CATE " + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    columns = ["Data", "Method", "Models", "Initial Disagreement", "Final Disagreement", "Initial Brier", "Final Brier",
               "Theorem 3.1", "T1", "T2", "Brier Scores"]
    results_df = pd.DataFrame(
        columns=columns).to_csv(results_csv_file_name, index=False)

    for i in tqdm(range(0,20)):
        train_1_df = []
        train_2_df = []
        test_dfs = []
        # train = []
        for group in df['leaf_node'].unique():
            df_group = df[df['leaf_node'] == group]
            train_group, test_group = train_test_split(df_group, train_size=0.5)
            train_1, train_2 = train_test_split(train_group, train_size=0.5)
            # Collect the splits
            train_1_df.append(train_1)
            train_2_df.append(train_2)
            # train.append(train_group)
            test_dfs.append(test_group)

        # Concatenate all group train/test splits
        df_train_1 = pd.concat(train_1_df).reset_index(drop=True)
        df_train_2 = pd.concat(train_2_df).reset_index(drop=True)
        df_test = pd.concat(test_dfs).reset_index(drop=True)
        # df_train = pd.concat(train).reset_index(drop=True)
        # t_learner_1 = BaseSRegressor(XGBRegressor(), control_name='control')
        t_learner_1 = UpliftTreeClassifier(control_name='control')
        t_learner_1.fit(X=df_train_1.drop(columns=['treatment', 'leaf_node', 'outcome'], axis=1).values,
                        treatment=df_train_1['treatment'], y=df_train_1['outcome'])
        # t_learner_2 = BaseSRegressor(XGBRegressor(), control_name='control')
        t_learner_2 = UpliftTreeClassifier(control_name='control')
        t_learner_2.fit(X=df_train_2.drop(columns=['treatment', 'leaf_node', 'outcome'], axis=1).values,
                        treatment=df_train_2['treatment'], y=df_train_2['outcome'])
        group_errors = []
        for group_number in df['leaf_node'].unique():
            X_group = df_test[df_test['leaf_node'] == group_number].drop(columns=['leaf_node', 'outcome', 'treatment'],
                                                                         axis=1)
            predicted_cate_1 = t_learner_1.predict(X=X_group.values).mean()
            predicted_cate_2 = t_learner_2.predict(X=X_group.values).mean()
            actual_cate = df_test[(df_test['leaf_node'] == group_number) & (df_test['treatment'] == 'treatment1')][
                              'outcome'].mean() - \
                          df_test[(df_test['leaf_node'] == group_number) & (df_test['treatment'] == 'control')][
                              'outcome'].mean()

            group_errors.append({
                'group': group_number,
                'actual_cate': actual_cate,
                'predicted_1': predicted_cate_1,
                'predicted_2': predicted_cate_2
            })
        group_df = pd.DataFrame(group_errors)
        data = CausalDataWrapper(group_df,'actual_cate','group')
        model1 = ModelWrapper(None, None, group_df['predicted_1'], None, None, True)
        model2 = ModelWrapper(None, None, group_df['predicted_2'], None, None, True)
        result_dict = {"Data": "Twin", "Method": "Different_Data", "Models": 'uplift tree'}
        reconcile_instance = Reconcile(model1, model2, data, 0.01, 0.04)
        reconcile_instance.reconcile(results_csv_file_name, result_dict)



# statlog_(german_credit_data)_144



