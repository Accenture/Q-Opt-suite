"""
Copyright (c) 2023 Objectivity Ltd.
"""

import logging
from typing import Any, cast

import azure.quantum.optimization as aqo  # type: ignore

from platforms.azure import Azure


class MSQIO(Azure):
    """
    The Platform class for the Microsoft Quantum Inspired Optimisation service,
    the actual solver used is configured using the solver -> `class` configuration
    in the MSQIO section of the yaml configuration file.

    WARNING: Microsoft has deprecated all annealer technology on Azure.
    """

    def __init__(self, config: dict) -> None:
        """
        :param config: the model section of the configuration file
        """
        super().__init__(config)

        if "solver" not in config or "class" not in config["solver"]:
            raise ValueError("MSQIO solver class not specified in config")
        self._solver_config = config["solver"]

    def solve(self, problem: Any, timeout: int, num_solutions_desired: int) -> Any:
        aqo_problem = cast(aqo.Problem, problem)
        assert num_solutions_desired == 1  # TODO implement multiple result support

        solver_class = getattr(aqo, self._solver_config["class"])
        config = {k: v for k, v in self._solver_config.items() if k != "class"}
        if timeout:
            config["timeout"] = timeout
        solver = solver_class(self._workspace, **config)

        logging.debug(
            "About to submit problem, solver=%s config=%s",
            self._solver_config["class"],
            config,
        )
        results = solver.optimize(aqo_problem)
        logging.info("MSQIO result cost=%f", results["solutions"][0]["cost"])
        return results

    def __str__(self) -> str:
        """
        :returns a string representation for logging purposes.
        """
        return f"MSQIO-{self._solver_config['class']}"
