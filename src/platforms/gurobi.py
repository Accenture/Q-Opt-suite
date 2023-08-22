"""
Copyright (c) 2023 Objectivity Ltd.
"""

from typing import Any, cast
import numpy as np
import scipy.sparse as ss  # type: ignore
import gurobipy as gp  # type: ignore
from qiskit_optimization import QuadraticProgram  # type: ignore
import qiskit_optimization.problems.variable as qv  # type: ignore
import qiskit_optimization.problems.linear_constraint as qlc  # type: ignore
from models.model import ModelStep

from platforms.platform import Platform

# pylint: disable=E1101
# See https://haominnn.medium.com/3aefa8ffae11


def _map_vartype(vartype: qv.VarType) -> gp.GRB:
    """
    Map Qiskit variable types to Gurobi variable types.

    :param vartype: the Qiskit variable type
    :returns the equivalent Gurobi variable type
    """
    if vartype == qv.VarType.BINARY:
        return gp.GRB.BINARY
    elif vartype == qv.VarType.INTEGER:
        return gp.GRB.INTEGER
    elif vartype == qv.VarType.CONTINUOUS:
        return gp.GRB.CONTINUOUS
    else:
        raise ValueError(f"Variable type {vartype} not supported in Gurobi")


def _map_sense(sense: qlc.ConstraintSense) -> str:
    """
    Map Qiskit constraint sense to Gurobi sense.

    :param sense: the Qiskit constraint sense
    :returns the equivalent Gurobi constraint sense `"=" ">="` or `"<="`
    """
    if sense == qlc.ConstraintSense.EQ:
        return "="
    elif sense == qlc.ConstraintSense.GE:
        return ">="
    elif sense == qlc.ConstraintSense.LE:
        return "<="
    else:
        raise ValueError(f"Sense {sense} not supported in Gurobi")


def _set_variables(problem: gp.Model, program: QuadraticProgram) -> None:
    """
    Set the variables in the `Model` according to the Qiskit `QuadraticProgram`.

    :param problem: the Gurobi `Model`
    :param program: the Qiskit `QuadraticProgram`
    """
    for variable in program.variables:
        problem.addVar(
            lb=variable.lowerbound,
            ub=float("inf") if variable.upperbound is None else variable.upperbound,
            vtype=_map_vartype(variable.vartype),
            name=variable.name,
        )


def _set_linear_constraints(problem: gp.Model, program: QuadraticProgram) -> None:
    """
    Set the linear constraints in the `Model` according to the Qiskit `QuadraticProgram`.

    :param problem: the Gurobi `Model`
    :param program: the Qiskit `QuadraticProgram`
    """
    # "Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient."
    # (Error message from scipy.) Indeed runtime with csr_array is excessive for larger instances
    # A = ss.csr_array((len(program.linear_constraints), len(program.variables)))
    constraint_matrix = ss.lil_array(
        (len(program.linear_constraints), len(program.variables))
    )
    sense = np.empty(len(program.linear_constraints), np.str_)
    rhs_vector = np.empty(len(program.linear_constraints))
    for c_index, constraint in enumerate(program.linear_constraints):
        for (_, x_index), coefficient in constraint.linear.coefficients.items():
            constraint_matrix[c_index, x_index] = coefficient
        sense[c_index] = _map_sense(constraint.sense)
        rhs_vector[c_index] = constraint.rhs

    problem.addMConstr(constraint_matrix, None, sense, rhs_vector)


def _set_quadratic_constraints(problem: gp.Model, program: QuadraticProgram) -> None:
    """
    Set the quadratic constraints in the `Model` according to the Qiskit `QuadraticProgram`.

    :param problem: the Gurobi `Model`
    :param program: the Qiskit `QuadraticProgram`
    """
    for constraint in program.quadratic_constraints:
        # Q = ss.csr_array((len(program.variables), len(program.variables)))
        constraint_matrix = ss.lil_array(
            (len(program.variables), len(program.variables))
        )
        for (
            x_index,
            y_index,
        ), coefficient in constraint.quadratic.coefficients.items():
            constraint_matrix[x_index, y_index] = coefficient

        constraint_vector = np.zeros(len(program.variables))
        for (_, x_index), coefficient in constraint.linear.coefficients.items():
            constraint_vector[x_index] = coefficient

        problem.addMQConstr(
            constraint_matrix,
            constraint_vector,
            _map_sense(constraint.sense),
            constraint.rhs,
        )


def _set_objective(problem: gp.Model, program: QuadraticProgram) -> None:
    """
    Set the objective in the `Model` according to the Qiskit `QuadraticProgram`.

    :param problem: the Gurobi `Model`
    :param program: the Qiskit `QuadraticProgram`
    """
    # Q = ss.csr_array((len(program.variables), len(program.variables)))
    objective_matrix = ss.lil_array((len(program.variables), len(program.variables)))
    for (
        x_index,
        y_index,
    ), coefficient in program.objective.quadratic.coefficients.items():
        objective_matrix[x_index, y_index] = coefficient

    objective_vector = np.zeros(len(program.variables))
    for (_, x_index), coefficient in program.objective.linear.coefficients.items():
        objective_vector[x_index] = coefficient

    problem.setMObjective(
        objective_matrix,
        objective_vector,
        program.objective.constant,
        sense=gp.GRB.MINIMIZE,
    )


class Gurobi(Platform):
    """The `Platform` implementation for Gurobi."""

    def translate_problem(self, step: ModelStep) -> gp.Model:
        """
        Translate the problem into a Gurobi `Model`.

        :param model: the `ModelStep` to translate
        :returns the model as a Gurobi `Model`
        """
        problem = gp.Model(str(step))
        _set_variables(problem, step.program)
        _set_linear_constraints(problem, step.program)
        _set_quadratic_constraints(problem, step.program)
        _set_objective(problem, step.program)
        return problem

    def num_variables(self, problem: Any) -> int:
        """
        :param problem: the Gurobi `Model`
        :returns the number of variabes in the `Model`
        """
        model = cast(gp.Model, problem)
        return model.NumVars

    def solve(self, problem: Any, timeout: int, num_solutions_desired: int) -> gp.Model:
        """
        Solves the `Model` and returns it as Gurobi embeds the results in the model.

        :param problem: the problem being solved as a `Model`
        :param timeout: the timeout in seconds
        :param num_solutions_desired: how many solutions should be returned
        :returns the `Model` now containing the optimised variable assignment
        """
        assert num_solutions_desired == 1  # TODO implement multiple result support
        model = cast(gp.Model, problem)

        if timeout:
            model.setParam("TimeLimit", timeout)
        model.write(f"gurobi-{model.NumVars}.mps")
        model.optimize()
        return model

    def get_info(self, result: Any) -> str:
        """
        Return any additional information about the result that should be logged.

        :param result: the Gurobi `Model` containing the optimised variable assignment
        :returns a string with solution information
        """
        model = cast(gp.Model, result)
        return f"status={model.Status} solutions={model.SolCount}"

    def get_solver_time(self, result: Any) -> float:
        """
        Return the actual solver time used in seconds.

        :param result: the Gurobi `Model` containing the optimised variable assignment
        :returns the solver time used
        """
        model = cast(gp.Model, result)
        return model.Runtime

    def translate_result(
        self,
        step: ModelStep,
        qubo: QuadraticProgram,
        result: Any,
        num_solutions_desired: int,
    ) -> list:
        """
        Translates the `Model` result into variable values for the `QuadraticProgram`.

        :param step: the `ModelStep` executing the optimisation problem
        :param qubo: the `QuadraticProgram` representing the model being solved
        :param result: the optimised variable assignment as a `Model`
        :param num_solutions_desired: how many solutions should be returned
        :returns the optimised variable assignment(s) in terms of the `QuadraticProgram`
        """
        assert num_solutions_desired == 1  # TODO implement multiple result support
        model = cast(gp.Model, result)

        return [v.X for v in model.getVars()]
