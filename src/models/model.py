"""
Copyright (c) 2023 Objectivity Ltd.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Callable, Optional
from numpy import ndarray

from qiskit_optimization import QuadraticProgram  # type: ignore
from qiskit_optimization.problems import VarType  # type: ignore
from models.qubo import (
    QuadraticProgramConverter,
    QuadraticProgramIdentityConverter,
    QuadraticProgramToQuboConverter,
    ToQuboCallback,
)


class ModelStep(ABC):
    """
    The ModelStep class abstracts a single step of the algorithm, or in simple cases this could
    represent the entire application for example the Traveling Salesman problem
    """

    @property
    def program(self) -> QuadraticProgram:
        return self._program

    @program.setter
    def program(self, program: QuadraticProgram) -> None:
        self._program = program

    @property
    def num_solutions_desired(self) -> int:
        return self._num_results_desired

    @abstractmethod
    def __init__(self) -> None:
        self.program = None
        self._num_results_desired = 1

    def is_qubo(self) -> bool:
        """
        Returns whether the step uses a pure QUBO formulation

        :returns `True` if the problem is both unconstrained and using binary variables only
        """

        def is_unconstrained():
            return not (
                self.program.linear_constraints or self.program.quadratic_constraints
            )

        def is_binary():
            for v in self.program.variables:
                if v.vartype != VarType.BINARY:
                    return False
            return True

        return is_unconstrained() and is_binary()

    def to_qubo(self, cb: ToQuboCallback, lagrange: Optional[float] = None) -> None:
        """
        Translates the step to a QUBO with a given (optional) Lagrange parameter for the
        constraints. This can be used by platforms which only support QUBOs.

        :param cb: the `ToQuboCallback` used to construct the QUBO formulation
        :param lagrange: the Lagrange parameter used for constraints. Defaults to 1
        """

        def create_converter() -> QuadraticProgramConverter:
            if self.is_qubo():
                logging.info(
                    f"Step with {len(self.program.variables)} variables is already a QUBO"
                )
                return QuadraticProgramIdentityConverter(self.program)
            else:
                logging.info(
                    f"Converting step with {len(self.program.variables)} variables to QUBO"
                )
                return QuadraticProgramToQuboConverter(self.program, lagrange)

        self.converter = create_converter()
        self.converter.convert(cb)

    def from_qubo(self, result: list, lagrange: Optional[float] = None) -> ndarray:
        """
        Translates the result of the step provided by to_qubo back to the original program.
        This can be used by platforms which only support QUBOs.

        :param result: the optimised variable assignment of the QUBO formulation
        :param lagrange: the Lagrange parameter used for constraints. Defaults to 1
        :returns the optimised variable assignment of the original non-QUBO problem
        """
        return self.converter.interpret(result)

    def is_qubo_sparse(self) -> bool:
        """
        Returns whether the QUBO translation of the problem has a sparse interaction matrix.

        :returns `True` if the QUBO interaction matrix is sparse.
            The default implementation returns `False` always.
        """
        return False

    def __str__(self) -> str:
        """
        Return a string representation of the model for logging purposes

        :return A human-readable name
        """
        return self.__class__.__name__


class Model(ABC):
    """
    The `Model` class abstracts a potentially more complex problem or application
    that may require multiple steps or iterations to solve.
    """

    def __init__(self, config: dict) -> None:
        """
        Instantiate an algorithm with the given configuration taken from the yaml file.
        Subclasses *must* maintain this signature.

        :param config: the model section of the configuration file
        """
        pass

    @abstractmethod
    def execute(self, step_callback: Callable[[ModelStep], list]) -> list:
        """
        Execute the model's algorithm.

        :param step_callback: used to execute the individual `ModelStep`s on the hardware platform
        :returns a list of variable values in the order used by the `QuadraticProgram`
        """
        raise NotImplementedError

    @abstractmethod
    def quality(self, result) -> float:
        """
        Measures result quality.

        :param result: an optimised variable assignment for the original model
        :returns a measure of the result quality.
        """
        raise NotImplementedError

    @abstractmethod
    def is_feasible(self, result) -> bool:
        """
        Determines whether the result is feasible.

        :param result: an optimised variable assignment for the original model
        :returns `True` if the result is feasible, that is, satisfies all the constraints
        """
        raise NotImplementedError


class SimpleModel(Model, ModelStep):
    """
    Convenience class for a `Model` which is implemented using a single step.
    """

    def __init__(self, config: dict) -> None:
        """
        Instantiate an algorithm with the given configuration taken from the yaml file.
        Subclasses *must* maintain this signature.

        :param config: the model section of the configuration file
        """
        Model.__init__(self, config)
        ModelStep.__init__(self)

    def execute(self, step_callback: Callable[[ModelStep], list]) -> list:
        return step_callback(self)

    def quality(self, result) -> float:
        """
        Returns a measure of the result quality by evaluating the problem Hamiltonian (without
        constraints), as given by the `program` variable, for the given solution.

        :param result: an optimised variable assignment for the original model
        :returns a measure of the result quality.
        """
        return self.program.objective.evaluate(result)

    def is_feasible(self, result) -> bool:
        """
        Returns whether the result is feasible, that is, satisfies all the constraints. This is
        done by simply delegating to the `QuadraticProgram` in the `program` variable.

        :param result: an optimised variable assignment for the original model
        :returns `True` if the result is feasible, that is, satisfies all the constraints
        """
        return self.program.is_feasible(result)
