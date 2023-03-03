import numpy as np
import random
import warnings
from typing import Callable, Union
from math import sqrt, ceil


def all_columns(X: np.array, rand: random.Random) -> range:
    """
    A function that returns a list of column indices considered for a split
    """
    return list(range(X.shape[1]))


def random_sqrt_columns(X: np.array, rand: random.Random) -> list:
    """
    Returns random sqrt number of columns of X, without repetition
    """
    all_cols = list(all_columns(X, rand))
    number_of_cols = ceil(sqrt(X.shape[1]))

    return list(rand.sample(all_cols, number_of_cols))


def find_best_split(X: np.array, 
                    y: np.array, 
                    candidate_columns: Union[list, range]) -> tuple:
    """
    Find best split and return attribute, which we want to split
    and value, by which we split
    """

    best_feature = None
    value_to_split = None
    lowest_gini = 1

    def gini_impurity(y):
        """
        Calculate Gini impurity of a set of target values.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        impurity = 1 - np.sum(probabilities**2)
        return impurity

    for feature in candidate_columns:
        values = X[:, feature]
        unique_values = np.unique(values)

        for value in unique_values:
            left_indices = X[:, feature] <= value
            right_indices = X[:, feature] > value

            left_y = y[left_indices]
            right_y = y[right_indices]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            left_impurity = gini_impurity(left_y)
            right_impurity = gini_impurity(right_y)

            gini = (len(left_y) / len(y)) * left_impurity + \
                                (len(right_y) / len(y)) * right_impurity

            if gini < lowest_gini:
                best_feature = feature
                value_to_split = value
                lowest_gini = gini

    return best_feature, value_to_split


class TreeNode:

    def __init__(self, 
                 leaf: bool, prediction=None,
                 attribute=None, value=None,
                 left_subtree=None, right_subtree=None):
        """
        if self.leaf == True: that means we are in a leaf and can direcly return predictions
        else: we are in a node, have to decide how to predict further
        """
        self.leaf = leaf
        self.prediction = prediction
        self.attribute = attribute
        self.value = value
        self.left_subtree = left_subtree
        self.right_subtree = right_subtree

    def predict(self, X: np.array) -> np.array:
        """
        Predicts values for attributes dataframe X
        """

        y_s = []

        for x in X:
            y_s.append(recursively_predict(x, self))

        return np.array(y_s)


def recursively_build(X: np.array, 
                      y: np.array, 
                      min_samples: int, 
                      candidates_columns_function: Callable, 
                      rand: random.Random) -> TreeNode:
    """
    Recursivelly builds a tree, consinsting of nodes
    """
    if len(X) >= min_samples:
        # split here
        candidate_columns = candidates_columns_function(X, rand)
        feature, value = find_best_split(X, y, candidate_columns)

        left_X = X[X[:, feature] <= value]
        left_y = y[X[:, feature] <= value]
        right_X = X[X[:, feature] > value]
        right_y = y[X[:, feature] > value]

        return TreeNode(False, attribute=feature, value=value,
                        left_subtree=recursively_build(left_X, left_y, min_samples, candidates_columns_function, rand),
                        right_subtree=recursively_build(right_X, right_y, min_samples, candidates_columns_function, rand))
    else:
        # too little samples, return prediction
        values, counts = np.unique(y, return_counts=True)

        if len(counts) < 2:
            return TreeNode(True, prediction=values[0])
        elif counts[0] > counts[1]:
            return TreeNode(True, prediction=values[0])
        elif counts[0] == counts[1]:
            warnings.warn("Cannot predict for sure. Same number of 0 and 1 samples. Predicted 0.")
            return TreeNode(True, prediction=values[1])
        else:
            return TreeNode(True, prediction=values[1])


def recursively_predict(x: np.array, tree: TreeNode):
    if tree.leaf:
        return tree.prediction
    else:
        if x[tree.attribute] <= tree.value:
            return recursively_predict(x, tree.left_subtree)
        else:
            return recursively_predict(x, tree.right_subtree)


class Tree:
    """
    Classification tree, which accepts input variable and predicts a binary target variable.
    """

    def __init__(self, rand: random.Random = None, 
                 get_candidate_columns: Callable = all_columns, 
                 min_samples: int=2) -> None:
        """
        Initiate a classification tree
        """
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples


    def build(self, X: np.array, y: np.array) -> TreeNode:
        """
        Build the tree.
        """
        return recursively_build(X, y, self.min_samples, self.get_candidate_columns, self.rand)


# class RandomForest:
# 
#     def __init__(self, rand=None, n=50):
#         self.n = n
#         self.rand = rand
#         self.rftree = Tree(...)  # initialize the tree properly
# 
#     # TODO: finish method build
#     def build(self, X, y):
#         # ...
#         return RFModel(...)

# 
# class RFModel:
# 
#     def __init__(self, ...):
#         # ...
# 
#     def predict(self, X):
#         # ...
#         return predictions
# 
#     def importance(self):
#         imps = np.zeros(self.X.shape[1])
#         # ...
#         return imps
# 
# 
# if __name__ == "__main__":
#     learn, test, legend = tki()
# 
#     print("full", hw_tree_full(learn, test))
#     print("random forests", hw_randomforests(learn, test))
# 