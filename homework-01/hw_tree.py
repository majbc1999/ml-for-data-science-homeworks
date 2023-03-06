import numpy as np
import random
import warnings
from typing import Callable, Union
from math import sqrt, floor


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
    number_of_cols = floor(sqrt(X.shape[1]))

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

            if gini <= lowest_gini:
                best_feature = feature
                value_to_split = value
                lowest_gini = gini

    return best_feature, value_to_split


def majority_class(x: np.array) -> int:
    num_zeros = np.sum(x == 0)
    num_ones = np.sum(x == 1)

    if num_zeros > num_ones:
        return 0
    elif num_ones > num_zeros:
        return 1
    else:
        warnings.warn("Same number of 0 and 1 samples. Predicted 0.")
        return 0


class TreeNode:

    def __init__(self,
                 leaf: bool, prediction=None,
                 attribute=None, value=None,
                 left_subtree=None, right_subtree=None):
        """
        if self.leaf == True: that means we are in a leaf and can direcly 
        return predictions.
        else: we are in a node, have to decide how to predict further.
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
                      candidates_columns_fun: Callable,
                      rand: random.Random) -> TreeNode:
    """
    Recursivelly builds a tree, consinsting of nodes
    """

    if len(np.unique(y)) == 1:
        return TreeNode(True, prediction=y[0])

    if len(X) >= min_samples:
        # split here
        candidate_columns = candidates_columns_fun(X, rand)
        feature, value = find_best_split(X, y, candidate_columns)

        if feature is None:
            return TreeNode(True, majority_class(y))

        left_X = X[X[:, feature] <= value]
        left_y = y[X[:, feature] <= value]
        right_X = X[X[:, feature] > value]
        right_y = y[X[:, feature] > value]

        return TreeNode(False, attribute=feature, value=value,
                        left_subtree=recursively_build(left_X,
                                                       left_y,
                                                       min_samples,
                                                       candidates_columns_fun,
                                                       rand),
                        right_subtree=recursively_build(right_X,
                                                        right_y,
                                                        min_samples,
                                                        candidates_columns_fun,
                                                        rand))
    else:
        # too little samples, return prediction
        values, counts = np.unique(y, return_counts=True)

        if len(counts) < 2:
            return TreeNode(True, prediction=values[0])
        elif counts[0] > counts[1]:
            return TreeNode(True, prediction=values[0])
        elif counts[0] == counts[1]:
            warnings.warn("Same number of 0 and 1 samples. Predicted 0.")
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
    Classification tree skeleton.
    """

    def __init__(self, rand: random.Random = None,
                 get_candidate_columns: Callable = all_columns,
                 min_samples: int = 2) -> None:
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
        return recursively_build(X, y, self.min_samples,
                                 self.get_candidate_columns, self.rand)


class RFModel:
    """
    An instance of classification trees, built on data, ready to predict
    """

    def __init__(self, trees: list[TreeNode], X: np.array, y: np.array):
        self.trees = trees
        self.X = X
        self.y = y

    def predict(self, X: np.array) -> np.array:
        predictions = []

        for tree in self.trees:
            vector_of_predictions = tree.predict(X)
            predictions.append(vector_of_predictions)

        predictions = np.array(predictions)
        final_predictions = []

        for i in range(len(predictions[0])):
            all_votes = predictions[:, i]
            final_predictions.append(majority_class(all_votes))

        return np.array(final_predictions)

    def importance(self) -> np.array:
        """
        Return an array of attribute importance. 
        Each element is importance of i-th attribute.
        """

        missclasified_original = 0
        N = len(self.trees) * len(self.X)

        for index, j in enumerate(self.X):
            for tree in self.trees:
                prediction = tree.predict(np.array([j]))[0]
                if prediction != self.y[index]:
                    missclasified_original += 1 / N

        missclasification_difference = []

        for i in range(len(self.X[0])):
            # create new_X, same as X, only with i-th column shuffled randomly
            new_X = np.copy(self.X)
            np.random.shuffle(new_X[:, i])

            missclasified_permuted = 0

            for tree in self.trees:
                for index, j in enumerate(new_X):
                    prediction = tree.predict(np.array([j]))
                    if prediction != self.y[index]:
                        missclasified_permuted += 1 / N

            missclasification_difference.append(
                missclasified_permuted - missclasified_original)

        return np.array(missclasification_difference)


class RandomForest:

    def __init__(self, rand: random.Random = None, n: int = 50):
        self.n = n
        self.rand = rand
        self.rftree = Tree(rand=self.rand,
                           get_candidate_columns=random_sqrt_columns,
                           min_samples=2)

    def build(self, X, y) -> RFModel:
        trees = []
        for _ in range(self.n):
            tree = self.rftree.build(X, y)
            trees.append(tree)
        return RFModel(trees, X, y)


###############################################################################
#                            PART 2: calculations                             #
###############################################################################

def import_dataset_to_np(path: str) -> tuple[np.array]:
    """
    Imports dataset from `.csv` file.

    Takes first n-1 columns as attributes and last column as target.
    Returns tuple of X, y
    """
    X = np.genfromtxt(path, delimiter=',')
    y = np.genfromtxt(path, delimiter=',', dtype=str)
    return (X[1:, :-1], y[1:, -1])


def return_train_and_test_data(path: str) -> tuple[np.array]:
    X, y = import_dataset_to_np(path)

    # Change to binary
    y = np.where(y == 'Bcr-abl', 1, 0)

    train = X[:130], y[:130]
    test = X[130:], y[130:]

    return train, test


def standard_error(y_true: np.array, y_pred: np.array) -> float:
    """
    Returns standard error between two vectors, `y_true` and `y_pred`
    """
    n = len(y_true)
    squared_error = np.sum((y_true - y_pred) ** 2)
    return np.sqrt(squared_error / (n - 1)) / np.sqrt(n)


def misclassification_rate(y_true: np.array, y_pred: np.array) -> float:
    """
    Returns missclasification rate for two vectors, `y_true` and `y_pred`
    """
    return np.mean(y_true != y_pred)


def hw_tree_full(train: tuple[np.array],
                 test: tuple[np.array]) -> tuple:
    """
    Returns missclassification rate, standard error for train and test data
    for Decision Trees training
    """
    train_X, train_y = train
    test_X, test_y = test

    tree = Tree()

    trained_tree = tree.build(train_X, train_y)
    pred_y_test = trained_tree.predict(test_X)
    pred_y_train = trained_tree.predict(train_X)

    # Missclassification for test data
    test_ms = misclassification_rate(test_y, pred_y_test)

    # Missclassification for train data
    train_ms = misclassification_rate(train_y, pred_y_train)

    train_se = standard_error(train_y, pred_y_train)
    test_se = standard_error(test_y, pred_y_test)

    return (train_ms, train_se), (test_ms, test_se)


def hw_randomforests(train: tuple[np.array],
                     test: tuple[np.array]) -> tuple:
    """
    Returns missclassification rate, standard error for train and test data
    for Decision Trees training
    """
    train_X, train_y = train
    test_X, test_y = test

    forest = RandomForest(rand=random.Random(), n=100)
    trained_forest = forest.build(train_X, train_y)

    pred_y_test = trained_forest.predict(test_X)
    pred_y_train = trained_forest.predict(train_X)

    # Missclassification for test data
    test_ms = misclassification_rate(test_y, pred_y_test)

    # Missclassification for train data
    train_ms = misclassification_rate(train_y, pred_y_train)

    train_se = standard_error(train_y, pred_y_train)
    test_se = standard_error(test_y, pred_y_test)

    return (train_ms, train_se), (test_ms, test_se)


def random_forests_importance(train: tuple[np.array]) -> dict:
    """
    Returns importance variables for 
    """
    train_X, train_y = train

    forest = RandomForest(rand=random.Random(), n=100)
    trained_forest = forest.build(train_X, train_y)

    return trained_forest.importance()


if __name__ == "__main__":
    train, test = return_train_and_test_data('homework-01/tki-resistance.csv')

    #print("full", hw_tree_full(train, test))
    #print("random forests", hw_randomforests(train, test))
    print("importance", random_forests_importance(train))
