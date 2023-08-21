"""
Copyright (c) 2023 Objectivity Ltd.
"""

from typing import Tuple
import numpy as np
import pandas as pd  # type: ignore

from sklearn.datasets import make_classification  # type: ignore
from sklearn.preprocessing import KBinsDiscretizer  # type: ignore
from qiskit_optimization import QuadraticProgram  # type: ignore

from models.model import SimpleModel


class FeatureSelection(SimpleModel):
    """
    A model for the feature selection problem as used in deep learning.
    """

    def __init__(self, config: dict) -> None:
        """
        :param config: the model section of the configuration file
        """
        super().__init__(config)

        # Set the number of features
        n_features = config["size"]
        # Generate the dataset
        X, y = generate_dataset(
            n_features,
            config.get("relevant_features", 0.2),
            config.get("n_samples", 1000),
            config.get("n_classes", 2),
            config.get("seed", 42),
            config.get("n_bins", 10),
        )

        # Prepare MIQUBO problem
        n_selected_features = int(n_features * config.get("selected_features", 0.1))
        self.program = miqubo_problem(X, y, n_selected_features)


def prob(data: np.ndarray) -> np.ndarray:
    """
    Compute probability distribution of the data.

    :param data: the data
    :returns the probability distribution
    """
    if len(data.shape) == 1:
        data = data[:, np.newaxis]

    joint_counts = pd.crosstab(*data.T)
    joint_prob = joint_counts / joint_counts.sum().sum()
    return joint_prob.values


def shannon_entropy(p: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy of a probability distribution.

    :param p: the probability distribution
    :returns an array with the Shannon entropy
    """
    p_nonzero = p[p > 0]
    return -np.sum(p_nonzero * np.log2(p_nonzero))


def conditional_shannon_entropy(p: np.ndarray, j: int) -> np.ndarray:
    """
    Compute conditional Shannon entropy of a probability distribution.

    :param p: the probability distribution
    :param j: the variable to compare to
    :returns an array with the conditional Shannon entropy
    """
    p_joint = np.copy(p)
    p_joint = np.divide(p_joint, p_joint.sum())

    axis_to_sum = tuple([i for i in range(len(p_joint.shape)) if i != j])
    p_conditional = np.sum(p_joint, axis=axis_to_sum)

    return shannon_entropy(p_joint) - shannon_entropy(p_conditional)


def mutual_information(p: np.ndarray, j: int) -> np.ndarray:
    """
    Compute mutual information between all variables and variable j.

    :param p: the probability distribution
    :param j: the variable to compare to
    :returns an array with the mutual entropy information
    """
    p_marginal = np.sum(p, axis=1 - j, keepdims=True)
    p_marginal /= p_marginal.sum()
    return shannon_entropy(p_marginal) - conditional_shannon_entropy(p, j)


def conditional_mutual_information(
    data: np.ndarray, j: int, *conditional_indices
) -> np.ndarray:
    """
    Compute conditional mutual information between variables X and variable Y conditional on variable Z.

    :param data: the data
    :param j:
    :param conditional_indices:
    :returns an array with the conditional mutual entropy information
    """
    marginal_conditional_indices = [i - 1 if i > j else i for i in conditional_indices]
    return conditional_shannon_entropy(
        np.sum(data, axis=j), *marginal_conditional_indices
    ) - conditional_shannon_entropy(data, j)


def cmi_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute conditional pairwise mutual information matrix.

    :param X: the data
    :param y: the labels
    :returns the matrix
    """
    n_features = X.shape[1]
    cmi_mat = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i + 1, n_features):
            data = np.column_stack((y, X[:, i], X[:, j]))
            cmi_mat[i, j] = conditional_mutual_information(data, 1, 2)

    return cmi_mat


def mi_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute mutual information scores between features and the target variable.

    :param X: the data
    :param y: the labels
    :returns a list of scores
    """
    n_features = X.shape[1]
    mi = np.zeros(n_features)

    for i in range(n_features):
        data = np.column_stack((y, X[:, i]))
        p = prob(data)
        mi_value = mutual_information(p, 1)
        mi[i] = mi_value

    return mi


def normalize_bqm(linear_terms: dict, quadratic_terms: dict) -> Tuple[dict, dict]:
    """
    Normalise the binary quadratic model.

    :param linear_terms: the linear BQM terms
    :param quadratic_terms: the quadratic BQM terms
    :returns a tuple with the normalised linear and quadratic terms
    """
    bias_range = (-1, 1)

    def min_and_max(iterable):
        if not iterable:
            return 0, 0
        return min(iterable), max(iterable)

    lin_min, lin_max = min_and_max(linear_terms.values())
    quad_min, quad_max = min_and_max(quadratic_terms.values())

    inv_scalar = max(
        lin_min / bias_range[0],
        lin_max / bias_range[1],
        quad_min / bias_range[0],
        quad_max / bias_range[1],
    )

    if inv_scalar != 0:
        normalized_linear = {
            key: coeff / inv_scalar for key, coeff in linear_terms.items()
        }
        normalized_quadratic = {
            key: coeff / inv_scalar for key, coeff in quadratic_terms.items()
        }
    else:
        normalized_linear = linear_terms
        normalized_quadratic = quadratic_terms

    return normalized_linear, normalized_quadratic


def miqubo_problem(
    X: np.ndarray, y: np.ndarray, n_selected_features: int
) -> QuadraticProgram:
    """
    Creates an MIQUBO problem using features, target variable, and desired number of selected features.

    :param X:
    :param y:
    :param n_selected_features: the number of features to select
    :returns the `QuadraticProgram`
    """
    n_features = X.shape[1]
    mi_scores_ = mi_scores(X, y)

    # Calculate conditional pairwise mutual information
    pairwise_cmi = cmi_matrix(X, y)

    # Create QUBO problem
    qubo = QuadraticProgram()
    for i in range(n_features):
        qubo.binary_var(f"x_{i}")

    # Create the linear terms dictionary
    linear_terms = {f"x_{i}": -1.0 * mi_scores_[i] for i in range(n_features)}

    # Create the quadratic terms dictionary
    quadratic_terms = {
        (f"x_{i}", f"x_{j}"): -1.0 * pairwise_cmi[i, j]
        for i in range(n_features)
        for j in range(n_features)
    }

    # Normalize the linear and quadratic terms
    normalized_linear, normalized_quadratic = normalize_bqm(
        linear_terms, quadratic_terms
    )

    # Update the QUBO problem
    qubo.minimize(linear=normalized_linear, quadratic=normalized_quadratic)

    # Add constraint: select exactly n_selected_features features
    qubo.linear_constraint(
        linear=list(np.ones(n_features)),
        sense="==",
        rhs=n_selected_features,
        name="select_features",
    )

    return qubo


def generate_dataset(
    n_features: int,
    relevant_features: int,
    n_samples: int,
    n_classes: int,
    random_state: int,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset with a specified number of features, samples, classes, random state, and bins.

    :param n_features: the number of features
    :param relevant_features: the number of features that are relevant
    :param n_samples: how many samples to generate
    :param n_classes: how many classes of data to generate
    :param random_state: the random number generator seed
    :param n_bins: the number of bins tu use when discretizing
    :returns a tuple with the discrete data X and labels y
    """
    n_informative = int(n_features * relevant_features)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        random_state=random_state,
    )

    # Discretize the continuous data into bins
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    X_discrete = discretizer.fit_transform(X)

    return X_discrete, y
