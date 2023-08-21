"""
Copyright (c) 2023 Objectivity Ltd.
"""

import random
import qiskit_optimization.applications.bin_packing as qabp  # type: ignore

from models.model import SimpleModel


class BinPacking(SimpleModel):
    """
    A model for the bin packing problem. This is just a thin wrapper around Qiskit's `bin_packing`.
    """

    def __init__(self, config: dict) -> None:
        """
        :param config: the model section of the configuration file
        """
        super().__init__(config)

        min_weight, max_weight = config.get("item_weight", [1, 100])
        max_bin_weight = config.get("max_bin_weight", max_weight)

        weights = [
            random.randint(min_weight, max_weight) for _ in range(config["size"])
        ]
        self.program = qabp.BinPacking(weights, max_bin_weight).to_quadratic_program()
