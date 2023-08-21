"""
Copyright (c) 2023 Objectivity Ltd.
"""

from typing import Any

import numpy as np
from qiskit_optimization import QuadraticProgram # type: ignore 
from models.model import ModelStep
from models.qubo import ToQuboNumpyCallback
from platforms.platform import Platform


class CSVReader(Platform):
    """
    A `Platform` implementation to read QUBO solutions from a CSV file and evaluate them.
    This only works for single-step algorithms.
    """

    def __init__(self, config: dict) -> None:
        """
        :param config: the model section of the configuration file
        """
        super().__init__(config)

        self._file = config.get("file")
        self._directory = config.get("directory")

    def translate_problem(self, step: ModelStep) -> np.ndarray:
        """
        Translate the problem into a numpy array

        :param step: the model's optimisation step just executed
        :returns the QUBO as a a numpy array
        """
        self._program = step.program
        self._name = f"{self._file or step}-{step.program.get_num_vars()}"
        cb = ToQuboNumpyCallback()
        self.construct_qubo(step, cb)
        return cb.interactions

    def num_variables(self, problem: np.ndarray) -> int:
        """
        :param problem: the QUBO as a a numpy array
        :returns the number of variabes in the problem
        """
        return len(problem)

    def solve(self, problem: Any, timeout: int, num_solutions_desired: int) -> list:
        """
        The reader solves nothing as such, it reads in the result CSV file.
        @returns the optimised variable assignment
        """
        directory = self._directory + "/" if self._directory else ""
        filename = f"results/{directory}{self._name}.txt"
        with open(filename, "r") as f:
            for line in f.readlines():
                if line.startswith(tuple("0123456789")):
                    return list(map(int, line.split(",")))
        
        raise RuntimeError(f"No data found in {filename}")

    def translate_result(
        self,
        step: ModelStep,
        qubo: QuadraticProgram,
        result: Any,
        num_solutions_desired: int
    ) -> np.ndarray:
        assert num_solutions_desired == 1
        return step.from_qubo(result)

    def __str__(self) -> str:
        """@returns a human-readble description of this `Platform` implementation"""
        return f"CSVReader-{self._file}"
