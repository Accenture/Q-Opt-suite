"""
Copyright (c) 2023 Objectivity Ltd.

Inspired by https://github.com/dwave-examples/3d-bin-packing
"""

from itertools import combinations, permutations
import logging
import random
from typing import Dict
from qiskit_optimization.problems import QuadraticProgram  # type: ignore
import numpy as np

from models.model import SimpleModel


class Cases:
    """
    Represents cuboid item data in a 3D bin packing problem.
    """

    def __init__(self, data):
        """
        :param data: dictionary containing raw information for both bins and cases
        """
        self.case_ids = np.repeat(data["case_ids"], data["quantity"])
        self.case_ids = np.repeat(data["case_ids"], data["quantity"])
        self.num_cases = np.sum(data["quantity"], dtype=np.int32)
        self.length = np.repeat(data["case_length"], data["quantity"])
        self.width = np.repeat(data["case_width"], data["quantity"])
        self.height = np.repeat(data["case_height"], data["quantity"])
        logging.debug("Number of cases: %d", self.num_cases)


class Bins:
    """
    Represents cuboid container data in a 3D bin packing problem.
    """

    def __init__(self, data, cases):
        """
        :param data: dictionary containing raw information for both bins and cases
        :param cases: Instance of ``Cases``, representing cuboid items packed into containers.
        """
        self.length = data["bin_dimensions"][0]
        self.width = data["bin_dimensions"][1]
        self.height = data["bin_dimensions"][2]
        self.num_bins = data["num_bins"]
        self.lowest_num_bin = np.ceil(
            np.sum(cases.length * cases.width * cases.height)
            / (self.length * self.width * self.height)
        )
        if self.lowest_num_bin > self.num_bins:
            raise RuntimeError(
                f"number of bins is at least {self.lowest_num_bin}, "
                + "try increasing the number of bins"
            )
        print(f"Minimum Number of bins required: {self.lowest_num_bin}")


class BinPacking3D(SimpleModel):  # pylint: disable=R0902
    """
    Model class for the 3D Binpacking problem
    https://github.com/dwave-examples/3d-bin-packing
    """

    def __init__(self, config: dict) -> None:
        """
        :param config: the model section of the configuration file
        """
        super().__init__(config)

        def partition(num_total, num_partitions):
            partitions = []
            for i in range(num_partitions - 1):
                partition = random.randint(
                    1, num_total - sum(partitions) - (num_partitions - i - 1)
                )
                partitions.append(partition)
            partitions.append(num_total - sum(partitions))
            return partitions

        num_cases = config["size"]
        num_sizes = num_cases // config.get("cases_per_size", 5)
        bin_dimensions = config.get("bin_dimensions", [50, 50, 50])

        data = {
            "num_bins": int(np.ceil(num_cases / config.get("size_per_bin", 10))),
            "bin_dimensions": bin_dimensions,
            "case_ids": list(range(num_sizes)),
            "quantity": partition(num_cases, num_sizes),
            "case_length": [
                random.randint(1, bin_dimensions[0]) for _ in range(num_sizes)
            ],
            "case_width": [
                random.randint(1, bin_dimensions[1]) for _ in range(num_sizes)
            ],
            "case_height": [
                random.randint(1, bin_dimensions[2]) for _ in range(num_sizes)
            ],
        }
        cases = Cases(data)
        bins = Bins(data, cases)

        self.program = QuadraticProgram("Binpacking3D")
        self._add_variables(cases, bins)
        effective_dimensions = self._add_orientation_constraints(cases)
        self._add_bin_on_constraint(bins, cases)
        self._add_geometric_constraints(bins, cases, effective_dimensions)
        self._add_boundary_constraints(bins, cases, effective_dimensions)
        self._define_objective(bins, cases, effective_dimensions)

    def _add_variables(self, cases: Cases, bins: Bins):
        num_cases = cases.num_cases
        num_bins = bins.num_bins
        self._x = self.program.integer_var_list(
            num_cases, lowerbound=0, upperbound=bins.length * bins.num_bins, name="x_"
        )
        self._y = self.program.integer_var_list(
            num_cases, lowerbound=0, upperbound=bins.width, name="y_"
        )
        self._z = self.program.integer_var_list(
            num_cases, lowerbound=0, upperbound=bins.height, name="z_"
        )
        self._bin_height = self.program.integer_var_list(
            num_bins, lowerbound=0, upperbound=bins.height, name="upper_bound_"
        )
        self._bin_loc = {
            (i, j): self.program.binary_var(name=f"case_{i}_in_bin_{j}")
            if num_bins > 1
            else 1
            for i in range(num_cases)
            for j in range(num_bins)
        }

        self._bin_loc = {
            (i, j): self.program.binary_var(name=f"case_{i}_in_bin_{j}")
            for i in range(num_cases)
            for j in range(num_bins)
        }
        self._bin_on = self.program.binary_var_list(
            num_bins, name="bin_", key_format="{}_is_used"
        )

        self._o = {
            (i, k): self.program.binary_var(name=f"o_{i}_{k}")
            for i in range(num_cases)
            for k in range(6)
        }

        self._selector = {
            (i, j, k): self.program.binary_var(name=f"sel_{i}_{j}_{k}")
            for i, j in combinations(range(num_cases), r=2)
            for k in range(6)
        }

    def _add_orientation_constraints(self, cases: Cases) -> list:
        num_cases = cases.num_cases
        d_x: Dict[int, Dict[str, int]] = {}
        d_y: Dict[int, Dict[str, int]] = {}
        d_z: Dict[int, Dict[str, int]] = {}
        for i in range(num_cases):
            perm_list = list(
                permutations([cases.length[i], cases.width[i], cases.height[i]])
            )
            d_x[i] = {}
            d_y[i] = {}
            d_z[i] = {}
            for j, (perm_a, perm_b, perm_c) in enumerate(perm_list):
                d_x[i] |= {self._o[i, j].name: perm_a}
                d_y[i] |= {self._o[i, j].name: perm_b}
                d_z[i] |= {self._o[i, j].name: perm_c}

        for i in range(num_cases):
            # one-hot constraint that each case can only be in one orientation at a time
            self.program.quadratic_constraint(
                linear={self._o[i, k].name: -1 for k in range(6)},
                quadratic={
                    (self._o[i, k].name, self._o[i, l].name): 2
                    for k in range(6)
                    for l in range(6)
                    if l > k
                },
                sense="==",
                rhs=-1,
                name=f"orientation_{i}",
            )
        return [d_x, d_y, d_z]

    def _add_bin_on_constraint(self, bins: Bins, cases: Cases):
        num_cases = cases.num_cases
        num_bins = bins.num_bins
        if num_bins > 1:
            for j in range(num_bins):
                self.program.quadratic_constraint(
                    linear={self._bin_loc[i, j].name: 1 for i in range(num_cases)},
                    quadratic={
                        (self._bin_on[j].name, self._bin_loc[i, j].name): -1
                        for i in range(num_cases)
                    },
                    sense="<=",
                    rhs=0,
                    name=f"bin_on_{j}",
                )
            for j in range(num_bins - 1):
                self.program.linear_constraint(
                    linear={self._bin_on[j].name: 1, self._bin_on[j + 1].name: -1},
                    sense=">=",
                    rhs=0,
                    name=f"bin_use_order_{j}",
                )

    def _add_geometric_constraints(
        self, bins: Bins, cases: Cases, effective_dimensions: list
    ):
        num_cases = cases.num_cases
        num_bins = bins.num_bins
        d_x, d_y, d_z = effective_dimensions

        for i, k in combinations(range(num_cases), r=2):
            # one-hot constraint
            self.program.quadratic_constraint(
                linear={self._selector[i, k, s].name: -1 for s in range(6)},
                quadratic={
                    (self._selector[i, k, s].name, self._selector[i, k, t].name): 2
                    for s in range(6)
                    for t in range(6)
                    if t > s
                },
                sense="==",
                rhs=-1,
                name=f"discrete_{i}_{k}",
            )
            for j in range(num_bins):
                cases_on_same_bin = (self._bin_loc[i, j].name, self._bin_loc[k, j].name)

                self.program.quadratic_constraint(
                    linear={
                        self._selector[i, k, 0].name: num_bins * bins.length,
                        self._x[i].name: 1,
                        self._x[k].name: -1,
                    }
                    | d_x[i],
                    quadratic={cases_on_same_bin: num_bins * bins.length},
                    sense="<=",
                    rhs=2 * num_bins * bins.length,
                    name=f"overlap_{i}_{k}_{j}_0",
                )

                self.program.quadratic_constraint(
                    linear={
                        self._selector[i, k, 1].name: bins.width,
                        self._y[i].name: 1,
                        self._y[k].name: -1,
                    }
                    | d_y[i],
                    quadratic={cases_on_same_bin: bins.width},
                    sense="<=",
                    rhs=2 * bins.width,
                    name=f"overlap_{i}_{k}_{j}_1",
                )

                self.program.quadratic_constraint(
                    linear={
                        self._selector[i, k, 2].name: bins.height,
                        self._z[i].name: 1,
                        self._z[k].name: -1,
                    }
                    | d_z[i],
                    quadratic={cases_on_same_bin: bins.height},
                    sense="<=",
                    rhs=2 * bins.height,
                    name=f"overlap_{i}_{k}_{j}_2",
                )

                self.program.quadratic_constraint(
                    linear={
                        self._selector[i, k, 3].name: num_bins * bins.length,
                        self._x[i].name: -1,
                        self._x[k].name: 1,
                    }
                    | d_x[k],
                    quadratic={cases_on_same_bin: num_bins * bins.length},
                    sense="<=",
                    rhs=2 * num_bins * bins.length,
                    name=f"overlap_{i}_{k}_{j}_3",
                )

                self.program.quadratic_constraint(
                    linear={
                        self._selector[i, k, 4].name: bins.width,
                        self._y[i].name: -1,
                        self._y[k].name: 1,
                    }
                    | d_y[k],
                    quadratic={cases_on_same_bin: bins.width},
                    sense="<=",
                    rhs=2 * bins.width,
                    name=f"overlap_{i}_{k}_{j}_4",
                )

                self.program.quadratic_constraint(
                    linear={
                        self._selector[i, k, 5].name: bins.height,
                        self._z[i].name: -1,
                        self._z[k].name: 1,
                    }
                    | d_z[k],
                    quadratic={cases_on_same_bin: bins.height},
                    sense="<=",
                    rhs=2 * bins.height,
                    name=f"overlap_{i}_{k}_{j}_5",
                )

        if num_bins > 1:
            for i in range(num_cases):
                # one-hot constraint
                self.program.quadratic_constraint(
                    linear={self._bin_loc[i, j].name: -1 for j in range(num_bins)},
                    quadratic={
                        (self._bin_loc[i, j].name, self._bin_loc[i, k].name): 2
                        for j in range(num_bins)
                        for k in range(num_bins)
                        if k > j
                    },
                    sense="==",
                    rhs=-1,
                    name=f"case_{i}_max_packed",
                )

    def _add_boundary_constraints(
        self, bins: Bins, cases: Cases, effective_dimensions: list
    ):
        num_cases = cases.num_cases
        num_bins = bins.num_bins
        d_x, d_y, d_z = effective_dimensions
        for i in range(num_cases):
            for j in range(num_bins):
                self.program.linear_constraint(
                    linear={
                        self._z[i].name: 1,
                        self._bin_height[j].name: -1,
                        self._bin_loc[i, j].name: bins.height,
                    }
                    | d_z[i],
                    sense="<=",
                    rhs=bins.height,
                    name=f"maxx_height_{i}_{j}",
                )

                self.program.linear_constraint(
                    linear={
                        self._x[i].name: 1,
                        self._bin_loc[i, j].name: num_bins * bins.length,
                    }
                    | d_x[i],
                    sense="<=",
                    rhs=bins.length * (j + 1 + num_bins),
                    name=f"maxx_{i}_{j}_less",
                )

                self.program.linear_constraint(
                    linear={
                        self._x[i].name: 1,
                        self._bin_loc[i, j].name: -bins.length * j,
                    },
                    sense=">=",
                    rhs=0,
                    name=f"maxx_{i}_{j}_greater",
                )

                self.program.linear_constraint(
                    linear={self._y[i].name: 1} | d_y[i],
                    sense="<=",
                    rhs=bins.width,
                    name=f"maxy_{i}_{j}_less",
                )

    def _define_objective(self, bins: Bins, cases: Cases, effective_dimensions: list):
        _, _, d_z = effective_dimensions

        first_obj_coefficient = 1.0
        second_obj_coefficient = 1.0
        third_obj_coefficient = 1.0

        # First term of objective: minimize average height of cases
        first_obj_term: Dict[str, float] = {}
        coeff = first_obj_coefficient / cases.num_cases
        for i in range(cases.num_cases):
            first_obj_term |= {self._z[i].name: coeff} | {
                k: v * coeff for k, v in d_z[i].items()
            }

        # Second term of objective: minimize height of the case at the top of the
        # bin
        second_obj_term: Dict[str, float] = {
            self._bin_height[j].name: second_obj_coefficient
            for j in range(bins.num_bins)
        }

        # Third term of the objective:
        third_obj_term: Dict[str, float] = {
            self._bin_on[j].name: bins.height * third_obj_coefficient
            for j in range(bins.num_bins)
        }

        self.program.minimize(linear=first_obj_term | second_obj_term | third_obj_term)
