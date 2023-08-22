"""
Copyright (c) 2023 Objectivity Ltd.
"""

from functools import partial
from itertools import chain
import logging
from typing import Any, List, Tuple, cast

import dimod  # type: ignore
from dwave.system import LeapHybridCQMSampler  # type: ignore

from qiskit_optimization import QuadraticProgram  # type: ignore
import qiskit_optimization.problems.variable as qv  # type: ignore
import qiskit_optimization.problems.linear_constraint as qlc  # type: ignore

from models.model import ModelStep
from models.qubo import EvaluateQuboCallback, QuadraticProgramToQuboConverter
from platforms.dwaveleap import DWaveLEAP


def _map_vartype(vartype: qv.VarType) -> dimod.Vartype:
    """
    Map Qiskit variable types to DWave variable types.

    @param vartype: the Qiskit type
    @returns the `dimod` type
    """
    if vartype == qv.VarType.BINARY:
        return dimod.BINARY
    elif vartype == qv.VarType.INTEGER:
        return dimod.INTEGER
    elif vartype == qv.VarType.CONTINUOUS:
        return dimod.REAL
    else:
        raise ValueError(f"Variable type {vartype} not supported in DWaveLEAP")


def _map_sense(sense: qlc.ConstraintSense) -> str:
    """
    Map Qiskit constraint senses to DWave senses.

    @param sense: the Qiskit sense
    @returns "==" or ">=" or "<="
    """
    if sense == qlc.ConstraintSense.EQ:
        return "=="
    elif sense == qlc.ConstraintSense.GE:
        return ">="
    elif sense == qlc.ConstraintSense.LE:
        return "<="
    else:
        raise ValueError(f"Sense {sense} not supported in DWaveLEAP")


def _convert_linear(
    variables: List[qv.Variable], item: Tuple[Tuple[int, int], float]
) -> Tuple[str, float]:
    """
    Convert a single Qiskit `LinearConstraint`
    into the (name, bias) tuple expected by DWave.

    :param variables: the `QuadraticProgram` variable list
    :param item: the `((index, index), coefficient)` tuple representing the constraint item
    :returns the 2-tuple encoding the constraint
    """
    (_, var), coefficient = item
    return (variables[var].name, coefficient)


def _convert_quadratic(variables, item) -> Tuple[str, str, float]:
    """
    Convert a single Qiskit `QuadraticConstraint`
    into the (name1, name2, bias) tuple expected by DWave.

    :param variables: the `QuadraticProgram` variable list
    :param item: the `((index1, index2), coefficient)` tuple representing the constraint item
    :returns the 3-tuple encoding the constraint
    """
    (var1, var2), coefficient = item
    return (variables[var1].name, variables[var2].name, coefficient)


def _set_variables(cqm: dimod.ConstrainedQuadraticModel, program: QuadraticProgram):
    """
    Set the variables in the `ConstraintQuadraticModel`
    according to the Qiskit `QuadraticProgram`.

    :param cqm: the D-Wave `ConstraintQuadraticModel`
    :param program: the Qiskit `QuadraticProgram`
    """
    for variable in program.variables:
        vartype = _map_vartype(variable.vartype)
        cqm.add_variable(
            vartype=vartype,
            v=variable.name,
            lower_bound=variable.lowerbound,
            upper_bound=variable.upperbound,
        )


def _set_linear_constraints(
    cqm: dimod.ConstrainedQuadraticModel, program: QuadraticProgram
):
    """
    Set the linear constraints in the `ConstraintQuadraticModel`
    according to the Qiskit `QuadraticProgram`.

    :param cqm: the D-Wave `ConstraintQuadraticModel`
    :param program: the Qiskit `QuadraticProgram`
    """
    for constraint in program.linear_constraints:
        coefficients = map(
            partial(_convert_linear, program.variables),
            constraint.linear.coefficients.items(),
        )
        cqm.add_constraint(
            coefficients,
            sense=_map_sense(constraint.sense),
            rhs=constraint.rhs,
            label=constraint.name,
        )


def _set_quadratic_constraints(
    cqm: dimod.ConstrainedQuadraticModel, program: QuadraticProgram
):
    """
    Set the quadratic constraints in the `ConstraintQuadraticModel`
    according to the Qiskit `QuadraticProgram`.

    :param cqm: the D-Wave `ConstraintQuadraticModel`
    :param program: the Qiskit `QuadraticProgram`
    """
    for constraint in program.quadratic_constraints:
        linear_coefficients = map(
            partial(_convert_linear, program.variables),
            constraint.linear.coefficients.items(),
        )
        quadratic_coefficients = map(
            partial(_convert_quadratic, program.variables),
            constraint.quadratic.coefficients.items(),
        )
        cqm.add_constraint(
            chain(linear_coefficients, quadratic_coefficients),
            sense=_map_sense(constraint.sense),
            rhs=constraint.rhs,
            label=constraint.name,
        )


def _set_objective(cqm: dimod.ConstrainedQuadraticModel, program: QuadraticProgram):
    """
    Set the objective in the `ConstraintQuadraticModel` according to the Qiskit `QuadraticProgram`.

    :param cqm: the D-Wave `ConstraintQuadraticModel`
    :param program: the Qiskit `QuadraticProgram`
    """
    linear_coefficients = map(
        partial(_convert_linear, program.variables),
        program.objective.linear.coefficients.items(),
    )
    quadratic_coefficients = map(
        partial(_convert_quadratic, program.variables),
        program.objective.quadratic.coefficients.items(),
    )
    cqm.set_objective(chain(linear_coefficients, quadratic_coefficients))


class DWaveCQM(DWaveLEAP):
    """
    The Platform class for the Constrained Quadratic Model sampler.
    """

    def __init__(self, config: dict) -> None:
        """
        Instantiate the D-Wave sampler with the given configuration taken from the yaml file.

        :param config: the model section of the configuration file
        """
        super().__init__(LeapHybridCQMSampler, config)
        self._calculate_qubo_energy = config.get("calculate_qubo_energy", False)

    def translate_problem(self, step: ModelStep) -> dimod.ConstrainedQuadraticModel:
        """
        Translate the problem into a D-Wave `ConstrainedQuadraticModel`.

        :param model: the `ModelStep` to translate
        :returns the model as a `BinaryQuadraticModel`
        """
        cqm = dimod.ConstrainedQuadraticModel()
        _set_variables(cqm, step.program)
        _set_linear_constraints(cqm, step.program)
        _set_quadratic_constraints(cqm, step.program)
        _set_objective(cqm, step.program)
        return cqm

    def num_variables(self, problem: Any) -> int:
        """
        Returns the number of variabes in the `ConstrainedQuadraticModel`.

        :param problem: the `ConstraintedQuadraticModel` representing the model being optimised
        :returns the number of variables in the model
        """
        cqm = cast(dimod.ConstrainedQuadraticModel, problem)
        return len(cqm.variables)

    def solve(
        self,
        problem: Any,
        timeout: int,
        num_solutions_desired: int,
    ) -> dimod.SampleSet:
        """
        Solves the `ConstrainedQuadraticModel` and returns the `SampleSet`.

        :param problem: the problem being solved as a `ConstraintedQuadraticModel`
        :param timeout: the timeout in seconds
        :param num_solutions_desired: how many solutions should be returned
        :returns the optimised variable assignment
        """
        cqm = cast(dimod.ConstrainedQuadraticModel, problem)
        assert num_solutions_desired == 1  # TODO implement multiple result support

        label = f"Dwave-CQM-{cqm.num_biases()}-{cqm.num_quadratic_variables()}"
        sampleset = self.sampler.sample_cqm(cqm, label=label, time_limit=timeout)
        feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
        is_feasible = len(feasible_sampleset.record) > 0
        sampleset = feasible_sampleset if is_feasible else sampleset
        logging.info(
            "DWave result energy=%f feasible=%s", sampleset.first.energy, is_feasible
        )
        return sampleset

    def translate_result(
        self,
        step: ModelStep,
        qubo: QuadraticProgram,
        result: Any,
        num_solutions_desired: int,
    ) -> list:
        """
        Translates the `SampleSet` into variable values for the `QuadraticProgram`.

        :param step: the `ModelStep` executing the optimisation problem
        :param qubo: the `QuadraticProgram` representing the model being solved
        :param sampleset: the optimised variable assignment of as a `SampleSet`
        :param num_solutions_desired: how many solutions should be returned
        :returns the optimised variable assignment(s) in terms of the `QuadraticProgram`.
        """
        assert num_solutions_desired == 1  # TODO implement multiple result support
        sampleset = cast(dimod.SampleSet, result)

        sample = sampleset.first.sample
        result = [sample[var.name] for var in step.program.variables]
        if self._calculate_qubo_energy:
            callback = EvaluateQuboCallback(result)
            QuadraticProgramToQuboConverter(step.program, self._lagrange).convert(
                callback
            )
            logging.info("Translated QUBO energy would be %f", callback.energy)

        return result
