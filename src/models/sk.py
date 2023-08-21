"""
Copyright (c) 2023 Objectivity Ltd.
"""

from qiskit_optimization.applications.sk_model import SKModel # type: ignore

from models.model import SimpleModel


class SK(SimpleModel):
    """
    A model for the Shannon-Kirkpatrick spin glass problem. This is just a thin wrapper around Qiskit's `sk_model`.
    """
    def __init__(self, config: dict) -> None:
        """
        :param config: the model section of the configuration file
        """
        super().__init__(config)

        skm = SKModel(num_sites=config["size"], rng_or_seed=0)
        self.program = skm.to_quadratic_program()
