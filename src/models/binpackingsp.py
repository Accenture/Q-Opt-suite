"""
Copyright (c) 2023 Objectivity Ltd.

Many thanks to Yaz Izumi of Toshiba.
"""

import logging
import random
from typing import Callable
import math
import numpy as np

from qiskit_optimization import QuadraticProgram  # type: ignore
from qiskit_optimization.problems.constraint import ConstraintSense  # type: ignore
import qiskit_optimization.applications.bin_packing as qabp  # type: ignore

from models.model import Model, ModelStep


class FindPackingsForFirstBin(ModelStep):
    """
    Runs the first step in the algorithm which is to find all the possible reasonable packings
    for the first bin. "Reasonable" means something between the lower and upper bounds.
    """

    def __init__(
        self, weights: list, lower_bound: int, upper_bound: int, num_candidates: int
    ) -> None:
        """
        :param weights: the weights of the items to pack
        :param lower_bound: the (soft) lower bound of the total item weight to pack in the container
        :param upper_bound: the (soft) upper bound of the total item weight to pack in the container
        :param num_candidates the power of 2 of the number of packings to look for, by default this
            is `2 ** len(weights)`
        """
        super().__init__()
        self.program = QuadraticProgram("BinpackingStep1")
        self._num_results_desired = math.ceil(2 ** (num_candidates or len(weights)))

        self.program.binary_var_list(len(weights))

        # Added the upper bound as a constraint to the original algorithm
        self.program.linear_constraint(
            linear=weights, sense=ConstraintSense.LE, rhs=upper_bound
        )

        # This Hamiltonian looks for values around the average of lower and upper bound...
        # H = (sum(xj * wj for (xj, wj) in zip(x, weights)) -
        #      (lower_bound + upper_bound) / 2)**2

        avg = (lower_bound + upper_bound) / 2

        # xj wj**2 - 2 * avg * sum xj wj
        linear = [wj**2 - 2 * avg for wj in weights]

        # 2 * sum_j<i xi wi xj wj
        quadratic = [
            [2 * wi * wj if j < i else 0 for j, wj in enumerate(weights)]
            for i, wi in enumerate(weights)
        ]

        # + avg ** 2
        constant = avg**2

        self._program.minimize(constant=constant, linear=linear, quadratic=quadratic)


class FindPackingSelection(ModelStep):
    """
    Runs the second step in the algorithm which is to try cover all the items with the packings
    found in the first step (i.e. a graph covering problem).
    """

    @property
    def candidates(self) -> list:
        """
        :returns the list of candidate packings
        """
        return self._candidates

    @candidates.setter
    def candidates(self, candidates: list) -> None:
        """
        :param candidates: the list of candidate packings to set
        """
        self._candidates = candidates

        self.program = QuadraticProgram("BinpackingStep1")

        self.program.binary_var_list(len(candidates))

        for i in range(self._num_weights):
            linear = [c[i] for c in candidates]
            if sum(linear):
                self.program.linear_constraint(
                    linear=linear, sense=ConstraintSense.EQ, rhs=1
                )
            else:
                logging.error("item %d is not allocated anywhere", i)

        linear = [1 for _ in range(len(candidates))]
        self.program.minimize(linear=linear)

    def __init__(self, weights: list) -> None:
        """
        :param weights: the weights of the items to pack
        """
        super().__init__()

        self._num_weights: int = len(weights)


class BinPackingSP(Model):
    """
    An heuristic implementation of the binpacking problem that is more friendly to annealers,
    as provided by Toshiba. Accepted configuration parameters are:

    - `candidates` the power of 2 of candidate packings in the first step. Default is `size`
        (i.e. `2^size` candidate packings)
    - `item_weight` the `[minimum, maximum]` weight of each item to pack
    - `max_bin_weight` the maximum weight accepted by each bin. Default is `item_weight[1]`
    """

    def __init__(self, config: dict) -> None:
        """
        :param config: the model section of the configuration file
        """
        super().__init__(config)

        min_weight, max_weight = config.get("item_weight", [1, 100])
        max_bin_weight: int = config.get("max_bin_weight", max_weight)
        self._weights = [
            random.randint(min_weight, max_weight) for _ in range(config["size"])
        ]
        num_candidates: int = int(config.get("candidates", len(self._weights)))

        self._find_packings_for_first_bin = FindPackingsForFirstBin(
            self._weights, max_bin_weight, max_bin_weight, num_candidates
        )
        self._find_packing_selection = FindPackingSelection(self._weights)

        # for verification and to calculate a comparable energy to standard binpacking
        self._program = qabp.BinPacking(
            self._weights, max_bin_weight
        ).to_quadratic_program()

    def execute(self, step_callback: Callable[[ModelStep], list]) -> list:
        # step 1, find all (near-)optimal packings of one bin
        self._find_packing_selection.candidates = list(
            filter(
                self._find_packings_for_first_bin.program.is_feasible,
                step_callback(self._find_packings_for_first_bin),
            )
        )

        # step 2, treat these candidate packings as a graph covering problem
        # and find a non-overlapping set of candidates packing all the items
        candidate_selection: list = step_callback(self._find_packing_selection)

        num_weights = len(self._weights)
        num_bins_used = 0

        occupancy = np.zeros(shape=(num_weights, num_weights), dtype=np.int32)
        remaining_items = np.ones(shape=(num_weights), dtype=np.int32)
        for is_selected, candidate in zip(
            candidate_selection, self._find_packing_selection.candidates
        ):
            if is_selected:
                occupancy[:, num_bins_used] = candidate
                remaining_items -= candidate
                num_bins_used += 1

        # the algorithm may leave a few items unallocated because the candidate
        # packings exclude suboptimal ones, so mop up any remaining items
        if np.sum(remaining_items):
            occupancy[:, num_bins_used] = remaining_items
            num_bins_used += 1

        return np.concatenate(
            (
                np.ones(shape=num_bins_used, dtype=np.int32),
                np.zeros(shape=num_weights - num_bins_used, dtype=np.int32),
                occupancy.reshape(occupancy.size),
            )
        )

    def quality(self, result) -> float:
        return self._program.objective.evaluate(result)

    def is_feasible(self, result) -> bool:
        return self._program.is_feasible(result)
