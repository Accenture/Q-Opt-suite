"""
Copyright (c) 2023 Objectivity Ltd.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union, cast
import numpy as np

from qiskit_optimization import QuadraticProgram # type: ignore

from models.model import ModelStep
from models.qubo import ToQuboCallback
from util.logging import BenchmarkLogger


class Platform(ABC):
    """
    The Platform class abstracts a single hardware platform, for example D-Wave LEAP.
    """

    @property
    def logger(self) -> BenchmarkLogger:
        return self._logger
    
    @logger.setter
    def logger(self, logger: BenchmarkLogger) -> None:
        self._logger = logger

    def __init__(self, config: dict) -> None:
        """
        Instantiate a platform with the given configuration taken from the yaml file

        :param config: the model section of the configuration file
        """
        self._lagrange = config.get("lagrange", None)

    def construct_qubo(self, model: ModelStep, cb: ToQuboCallback) -> None:
        """
        Turn the model into a QUBO using the configured lagrange parameter.
        The default implementation simply delegates to `model.to_qubo`.

        :param model: the `ModelStep` whose model needs to be converted into a QUBO
        :param cb: the `ToQuboCallback` used to construct the QUBO
        """
        model.to_qubo(cb, lagrange=self._lagrange)

    def translate_problem(self, step: ModelStep) -> Any:
        """
        Translate the problem into the platform-specific format. This default implementation
        simply returns the Qiskit QuadraticProgram contained in the step. This translation
        step is not included in the benchmark timing so as not to privilege Qiskit-native platforms.
        The default implementation returns the QuadraticProgram contained in the model.

        :param model: the `ModelStep` to translate into the platform's native format
        :returns the model in a platform-native format
        """
        return step.program

    def num_variables(self, problem: Any) -> int:
        """
        Retrieve the number of variables used in the platform-specific problem formulation.
        The default implementation returns the number of variables in a QuadraticProgram.

        :param problem: the representation the model being optimised. The default implementation
            assumes this to be a `QuadraticProgram`
        :returns the number of variables in the model
        """
        return len(cast(QuadraticProgram, problem.program.variables))

    @abstractmethod
    def solve(self, problem: Any, timeout: int, num_solutions_desired: int) -> Any:
        """
        Solves the problem (supplied in the platform-specific format) and returns the result
        in a platform-specific format. This is the core benchmarked function.

        :param problem: the problem being solved in a platform-specific format
        :param timeout: the timeout in seconds
        :param num_solutions_desired: how many solutions should be returned
        :returns the optimised variable assignment of the platform-specific problem
        """
        raise NotImplementedError

    def get_info(self, result: Any) -> Optional[str]:
        """
        Return any additional information about the result that should be logged. The default
        implementation returns `None`.

        :param result the optimised variable assignment of the platform-specific problem
        :returns a string with information about the problem's execution
        """
        return None

    def get_solver_time(self, result: Any) -> Optional[float]:
        """
        Find the actual solver time used in seconds. The default implementation returns
        `None` to indicate that this stat isn't available for the platform.

        :param result: the optimised variable assignment of the platform-specific problem
        :returns the execution time in seconds
        """
        return None

    def get_cost(self, result: Any) -> Optional[float]:
        """
        Finds the monetary cost of the job. The default implementation returns `None` to
        indicate that this stat isn't available for the platform.

        :param result: the optimised variable assignment of the platform-specific problem
        :returns a floating point number representing the cost, or `None`
        """
        return None

    def translate_result(
        self,
        step: ModelStep,
        qubo: QuadraticProgram,
        result: Any,
        num_solutions_desired: int,
    ) -> Union[list, np.ndarray]:
        """
        Translate the result given in the platform-specific format back into a list of variable
        values, in the order used by the `step.program` `QuadraticProgram`. This translation step
        is not included in the benchmark timing so as not to privilege Qiskit-native platforms.
        The default implementation simply returns the result, assuming it is singular.

        :param step: the `ModelStep` executing the optimisation problem
        :param qubo: the `QuadraticProgram` representing the model being solved
        :param result: the optimised variable assignment of the platform-specific problem
        :param num_solutions_desired: how many solutions should be returned
        :returns the optimised variable assignment(s) in terms of the `QuadraticProgram`.
            If `num_solutions_desired == 1` this is an array-like with the variable assignements;
            otherwise it is a list (of size 0, 1 or more) of array-like assignments.
        """
        assert num_solutions_desired == 1
        return result
