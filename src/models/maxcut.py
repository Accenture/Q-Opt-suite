"""
Copyright (c) 2023 Objectivity Ltd.
"""

from networkx.generators import gnp_random_graph  # type: ignore
import qiskit_optimization.applications.max_cut as qamc  # type: ignore

from models.model import SimpleModel


class Maxcut(SimpleModel):
    """
    A model for the max-cut problem. This is just a thin wrapper around Qiskit's `max_cut`.
    """

    def __init__(self, config: dict) -> None:
        """
        :param config: the model section of the configuration file
        """
        super().__init__(config)

        graph = gnp_random_graph(config["size"], config.get("p", 0.5))
        self.program = qamc.Maxcut(graph).to_quadratic_program()
