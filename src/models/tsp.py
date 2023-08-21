"""
Copyright (c) 2023 Objectivity Ltd.
"""

import qiskit_optimization.applications.tsp as tsp # type: ignore

from models.model import SimpleModel


class TSP(SimpleModel):
    """
    A model for the traveling salesman problem. This is just a thin wrapper around Qiskit's `tsp`.
    """

    def __init__(self, config: dict) -> None:
        """
        :param config: the model section of the configuration file
        """
        super().__init__(config)

        problem = tsp.Tsp.create_random_instance(
            n=config["size"], seed=config.get("seed", 123)
        )
        self.program = problem.to_quadratic_program()
