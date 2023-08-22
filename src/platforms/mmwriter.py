"""
Copyright (c) 2023 Objectivity Ltd.
"""

from array import array
from io import BytesIO
from itertools import repeat
import logging
from typing import Any, Optional, TextIO, cast
import numpy as np
import scipy.sparse as ss  # type: ignore
from scipy.io import mmwrite  # type: ignore

from qiskit_optimization import QuadraticProgram  # type: ignore
from qiskit_optimization.problems.constraint import ConstraintSense  # type: ignore

from models.model import ModelStep
from models.qubo import ToQuboNumpyCallback
from platforms.platform import Platform

CONSTRAINT_SENSES = {
    ConstraintSense.EQ: "==",
    ConstraintSense.LE: "<=",
    ConstraintSense.GE: ">=",
}


def dump_program(file: TextIO, program: QuadraticProgram) -> None:
    """
    Dump a somewhat human-readable description of the problem as described by the
    `QuadraticProgram`.

    :param file: the file handle to write the dump to
    :param program: the `QuadraticProgram` to dump
    """
    num_vars = program.get_num_vars()
    threshold = num_vars / 3

    def dump_coordinate(args):
        (var1, var2), coefficient = args
        return f"({var1}, {var2}, {coefficient})"

    def dump_array_full(coefficients):
        interactions = array("f", repeat(0.0, num_vars))
        for (_, var), coefficient in coefficients.items():
            interactions[var] += coefficient
        file.write(f"[{', '.join(map(str, interactions))}]")

    def dump_array_coordinates(coefficients):
        file.write(f"[{', '.join(map(dump_coordinate, coefficients.items()))}]")

    def dump_array(coefficients):
        is_dense = len(coefficients) > threshold
        dump = dump_array_full if is_dense else dump_array_coordinates
        dump(coefficients)

    def dump_matrix_full(linear_coefficients, quadratic_coefficients):
        interactions = [array("f", repeat(0.0, num_vars)) for _ in range(num_vars)]
        for (_, var), coefficient in linear_coefficients.items():
            interactions[var][var] += coefficient

        for (var1, var2), coefficient in quadratic_coefficients.items():
            interactions[var1][var2] += coefficient

        file.write("[")
        for var in range(num_vars):
            file.write(f"[{', '.join(map(str, interactions[var]))}]\n")
            if var < num_vars - 1:
                file.write(",\n")
        file.write("]")

    def dump_matrix_coordinates(linear_coefficients, quadratic_coefficients):
        file.write(
            f"[{', '.join(map(dump_coordinate, linear_coefficients.items()))},"
            f" {', '.join(map(dump_coordinate, quadratic_coefficients.items()))}]"
        )

    def dump_matrix(linear_coefficients, quadratic_coefficients):
        is_dense = (
            len(quadratic_coefficients) + len(linear_coefficients)
            > threshold * num_vars
        )
        dump = dump_matrix_full if is_dense else dump_matrix_coordinates
        dump(linear_coefficients, quadratic_coefficients)

    def dump_constraint(constraint):
        dump_array(constraint.linear.coefficients)
        file.write(f" {CONSTRAINT_SENSES[constraint.sense]} {constraint.rhs}\n")

    if len(program.objective.quadratic.coefficients):  # quadratic objective
        file.write("Objective (quadratic):\n")
        dump_matrix(
            program.objective.linear.coefficients,
            program.objective.quadratic.coefficients,
        )
        file.write("\n")
    else:  # linear terms only
        file.write("Objective (linear):\n")
        dump_array(program.objective.linear.coefficients)
        file.write("\n")

    file.write("\nConstraints:\n")
    for constraint in program.linear_constraints:
        dump_constraint(constraint)


class MMWriter(Platform):
    """A `Platform` class to write QUBOs to MatrixMarket files."""

    def __init__(self, config: dict) -> None:
        """
        :param config: the model section of the configuration file
        """
        super().__init__(config)

        self._file = config.get("file")
        self._program: Optional[QuadraticProgram] = None
        self._name: Optional[str] = None

    def translate_problem(self, step: ModelStep) -> np.ndarray:
        """
        Translate the problem into a numpy matrix.

        :returns the interaction matrix
        """
        self._program = step.program
        self._name = f"{self._file or step}-{step.program.get_num_vars()}"
        callback = ToQuboNumpyCallback()
        self.construct_qubo(step, callback)
        return callback.interactions

    def num_variables(self, problem: Any) -> int:
        """
        :returns the number of variabes in the problem.
        """
        interactions = cast(np.ndarray, problem)
        return len(interactions)

    def solve(self, problem: Any, timeout: int, num_solutions_desired: int) -> None:
        assert num_solutions_desired == 1
        interactions = cast(np.ndarray, problem)

        # use a sparse coordinate format when appropriate
        nonzero_interactions = np.count_nonzero(interactions)
        is_sparse = nonzero_interactions / interactions.size < 0.2  # heuristic
        logging.debug(
            "problem has %d/%d nonzero interactions (sparse=%s)",
            nonzero_interactions,
            interactions.size,
            is_sparse,
        )
        if is_sparse:
            interactions = ss.coo_array(interactions)

        # convert into a symmetric matrix
        problem_symm = (interactions + interactions.T) / 2

        # write the problem's interaction matrix as a MatrixMarket file
        with open(
            f"{self.logger.directory}/{self._name}.mm", "w", encoding="utf-8"
        ) as file:
            target = BytesIO()
            mmwrite(target, problem_symm, field="real", symmetry="symmetric")
            file.write(target.getvalue().decode())

        # write a higher-level description of the problem
        with open(
            f"{self.logger.directory}/{self._name}.txt", "w", encoding="utf-8"
        ) as file:
            dump_program(file, self._program)

    def translate_result(
        self,
        step: ModelStep,
        qubo: QuadraticProgram,
        result: Any,
        num_solutions_desired: int,
    ) -> list:
        """
        Dummy implementation because all we're doing is writing a file.

        :returns a variable assignment with 0 for every variable
        """
        assert num_solutions_desired == 1

        return [0 for _ in range(step.program.get_num_vars())]

    def __str__(self) -> str:
        """
        :returns a string representation for logging purposes.
        """
        return f"MMWriter-{self._file}"
