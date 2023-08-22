"""
Copyright (c) 2023 Objectivity Ltd.
"""

from abc import abstractmethod
from collections import namedtuple
from itertools import repeat
import logging
from typing import Any, List, Optional, Union, cast
from math import log2, ceil
from array import array
import numpy as np

from qiskit_optimization import QuadraticProgram  # type: ignore
from qiskit_optimization.problems.variable import Variable  # type: ignore
from qiskit_optimization.problems.constraint import Constraint, ConstraintSense  # type: ignore


class ToQuboCallback:
    """
    Callback interface used by the hardware platform implementation to construct a QUBO
    in whatever the platform-specific format is.

    In the QUBO, variables are numbered sequentially.
    """

    def set_num_variables(self, num_variables: int) -> None:
        """
        Set the number of binary variables in the QUBO including mapped integers and slack
        variables. The "names" of the variables will be `range(num_variables)`. The default
        implementation does nothing.
        """

    @abstractmethod
    def add_constant(self, constant: float) -> None:
        """
        Add a constant term to the QUBO. This is an adder, not a setter, in other words
        the coefficient will be added to any earlier value that may have been set.
        """
        raise NotImplementedError

    def add_linear(self, var: int, coefficient: float) -> None:
        """
        Add a linear term to the QUBO. This is an adder, not a setter, in other words the
        coefficient will be added to any earlier value that may have been set.

        The default implementation simply calls `add_quadratic(var, var, coefficient)` but
        the platform-specific implementation can override this behaviour.
        """
        self.add_quadratic(var, var, coefficient)

    @abstractmethod
    def add_quadratic(self, var1: int, var2: int, coefficient: float) -> None:
        """
        Add a quadratic term to the QUBO. This is an adder, not a setter, in other words
        the coefficient will be added to any earlier value that may have been set.
        """
        raise NotImplementedError


class ToQuboDeduplicatingCallback(ToQuboCallback):
    """
    Callback interface used by the hardware platform implementation to construct a QUBO
    in whatever the platform-specific format is. This version internally adds up the
    different contributions to the terms before calling the platform-specific API; it is
    intended for APIs that do not gracefully handle adding to terms and therefore prefer
    set-type calls.

    This implementation assumes that the interaction matrix is not extremely sparse.
    """

    def __init__(self, is_lower_half: bool = True) -> None:
        super().__init__()
        self.is_lower_half = is_lower_half
        self.constant = 0.0
        self.interactions: list[array] = []

    def set_num_variables(self, num_variables: int) -> None:
        """Set the number of binary variables in the QUBO."""
        self.constant = 0.0
        # The interaction matrix is a num_variabes * num_variabes list of array objects
        self.interactions = [
            array("f", repeat(0.0, num_variables)) for _ in range(num_variables)
        ]

    def add_constant(self, constant: float) -> None:
        self.constant += constant

    def add_quadratic(self, var1: int, var2: int, coefficient: float) -> None:
        """Add a quadratic term to the interaction matrix."""
        if (var2 > var1) == self.is_lower_half:
            var1, var2 = var2, var1
        self.interactions[var1][var2] += coefficient

    @abstractmethod
    def set_constant(self, constant: float) -> None:
        """Sets the constant offset of the QUBO."""
        raise NotImplementedError

    @abstractmethod
    def set_linear(self, var: int, coefficient: float) -> None:
        """Sets a linear term in the QUBO."""
        raise NotImplementedError

    @abstractmethod
    def set_quadratic(self, var1: int, var2: int, coefficient: float) -> None:
        """Sets a quadratic term in the QUBO."""
        raise NotImplementedError

    def set_interactions(self) -> None:
        """
        Assembles the final QUBO by calling the `set_linear` and `set_quadratic` methods.
        Alternatively, the client could could directly process the `self.interactions`
        matrix and `self.constant`.

        In order to maintain memory efficiency **this method is destructive** and consumes
        the interaction matrix during its execution.
        """
        logging.debug("offset=%f", self.constant)
        for interaction in self.interactions:
            logging.debug("\t".join(map(str, interaction)))

        self.set_constant(self.constant)

        for i in range(len(self.interactions) - 1, -1, -1):
            for j, coefficient in enumerate(
                self.interactions.pop()
            ):  # popping to release memory
                if coefficient != 0.0:
                    if i == j:
                        self.set_linear(i, coefficient)
                    else:
                        self.set_quadratic(i, j, coefficient)


class ToQuboNumpyCallback(ToQuboCallback):
    """QUBO converter callback function which populates a Numpy matrix `self.interactions`"""

    def __init__(self, is_lower_half: bool = True) -> None:
        super().__init__()
        self.is_lower_half = is_lower_half
        self.constant = 0.0
        self.interactions: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(
            [0, 0], dtype=np.float64
        )

    def set_num_variables(self, num_variables: int) -> None:
        self.constant = 0.0
        self.interactions = np.zeros([num_variables, num_variables], dtype=np.float64)

    def add_constant(self, constant: float) -> None:
        self.constant += constant

    def add_quadratic(self, var1: int, var2: int, coefficient: float) -> None:
        if (var2 > var1) == self.is_lower_half:
            var1, var2 = var2, var1
        self.interactions[var1][var2] += coefficient


class QuadraticProgramConverter:
    """
    Abstract base class for converters that turn a QuadraticProgram into a QUBO with
    the help of a callback function (to avoid the memory overhead of an intermediate
    representation)
    """

    def __init__(self, program: QuadraticProgram) -> None:
        """
        Construct a converter for the given `QuadraticProgram`.
        """
        self.program: QuadraticProgram = program

    @abstractmethod
    def convert(self, callback: ToQuboCallback) -> None:
        """
        Convert the encapsulated QuadraticProgram to a QUBO using the callback.
        """
        raise NotImplementedError

    @abstractmethod
    def interpret(self, result: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Interpret a QUBO result back into a result for the original `QuadraticProgram`
        using the conversion information.
        """
        raise NotImplementedError

    @abstractmethod
    def num_variables(self) -> int:
        """Returns the number of binary variables used in the converted QUBO."""
        raise NotImplementedError


class QuadraticProgramIdentityConverter(QuadraticProgramConverter):
    """
    A trivial "converter" used when the QuadraticProgram is already in QUBO format.
    """

    def convert(self, callback: ToQuboCallback) -> None:
        """
        Constructs a platform-specific QUBO directly from the information in the
        `QuadraticProgram` as the latter is already a QUBO
        """
        callback.set_num_variables(len(self.program.variables))

        callback.add_constant(self.program.objective.constant)

        for (
            _,
            var_x,
        ), coefficient in self.program.objective.linear.coefficients.items():
            callback.add_linear(var_x, coefficient)

        for (
            var_x,
            var_y,
        ), coefficient in self.program.objective.quadratic.coefficients.items():
            if var_x != var_y:
                callback.add_quadratic(var_x, var_y, coefficient)
            else:  # the Qiskit QuadraticProgramToQubo converter commits this horror
                callback.add_linear(var_x, coefficient)

    def interpret(self, result: Union[np.ndarray, list[float]]) -> np.ndarray:
        """Returns `result`. The QUBO results need no interpretation in this identity mapping"""
        return cast(np.ndarray, result)

    def num_variables(self) -> int:
        """Returns the number of variables in the QUBO, identical to the `QuadraticProgram`"""
        return len(self.program.variables)


# Structured tuple storing information about the mapping of a problem variable to QUBO
#
# - `index` the index of the variable in the QUBO model
# - `bits` the number of bits occupied in the QUBO model
# - `msb_coef` the coefficient of the most significant bit (the others are `2**bit_position`)
# - `upperbound` the upper bound (inclusive) of the variable
# - `lowerbound` the lower bound (inclusive) of the variable
VariableMappingInfo = namedtuple(
    "VariableMappingInfo", ["index", "bits", "msb_coef", "upperbound", "lowerbound"]
)


def coeff(variable: VariableMappingInfo, bit: int) -> float:
    """
    Return the value of the given bit in the variable. This is `2**bit_position` except
    for the most significant bit which has a value to match the range of the variable.
    """
    return float(2**bit) if bit < variable.bits - 1 else variable.msb_coef


class QuadraticProgramToQuboConverter(QuadraticProgramConverter):
    """
    A converter to turn a general QuadraticProgram into a platform-specific QUBO using
    a callback function to avoid intermediate representations.

    WARNING only the features required by our benchmarks are implemented; an error will
    be thrown if unimplemented features are used.
    """

    def __init__(self, program: QuadraticProgram, lagrange: Optional[float]) -> None:
        """
        Construct a converter for the given `QuadraticProgram` and Lagrange parameter.
        At the moment only a single Lagrange parameter is supported.
        """
        super().__init__(program)

        self.lagrange: float = float(lagrange) if lagrange else 1.0
        self.num_qubo_vars: int = 0

        def allocate_variable(
            upperbound: int, lowerbound: int = 0
        ) -> VariableMappingInfo:
            # Allocate QUBO variable space for a variable with the given bounds
            num_bits = ceil(
                log2(upperbound - lowerbound + 1)
            )  # upperbound is inclusive
            # The coeff of the most significant bit is tuned to produce exactly the variable range
            msb_coef = float(upperbound - lowerbound) - (2 ** (num_bits - 1) - 1)
            var = VariableMappingInfo(
                self.num_qubo_vars, num_bits, msb_coef, upperbound, lowerbound
            )
            self.num_qubo_vars += num_bits
            return var

        def allocate_problem_variable(variable: Variable) -> VariableMappingInfo:
            # Allocate QUBO variable space for the given problem variable
            if variable.vartype == Variable.Type.BINARY:
                var = allocate_variable(1)
                logging.debug(
                    "adding binary problem variable %s at position %d",
                    variable.name,
                    var.index,
                )
            elif variable.vartype == Variable.Type.INTEGER:
                var = allocate_variable(variable.upperbound, variable.lowerbound)
                logging.debug(
                    "adding %d bit integer problem variable %s at position %d",
                    var.bits,
                    variable.name,
                    var.index,
                )
            else:
                raise RuntimeError(f"Variable type {variable.vartype} not implemented")
            return var

        def allocate_slack_variable(
            constraint: Constraint,
        ) -> Optional[VariableMappingInfo]:
            # Allocate QUBO variable space for the given slack variable. There are some special
            # inequality constraint forms that do not require a slack variable but as they don't
            # occur in our models we're making no effort to detect those.
            if constraint.sense == ConstraintSense.LE:
                lowerbound = (
                    constraint.linear.bounds.lowerbound
                )  # the lowerbound of the constraint left hand side
                var = allocate_variable(constraint.rhs - lowerbound)
            elif constraint.sense == ConstraintSense.GE:
                upperbound = (
                    constraint.linear.bounds.upperbound
                )  # the upperbound of the constraint left hand side
                var = allocate_variable(upperbound - constraint.rhs)
            elif constraint.sense == ConstraintSense.EQ:
                return None  # No slack variable required for this constraint
            else:
                raise RuntimeError(
                    f"Constraint sense {constraint.sense} not implemented"
                )
            logging.debug(
                "adding slack variable of %d bits for %s at position %d",
                var.bits,
                constraint.name,
                var.index,
            )
            return var

        # Create the QUBO variable mapping for the `QuadraticProgram`
        self.problem_variables = list(map(allocate_problem_variable, program.variables))
        self.slack_variables = list(
            map(allocate_slack_variable, program.linear_constraints)
        )
        if program.quadratic_constraints:
            raise RuntimeError("Quadratic constraints not implemented")

    def convert(self, callback: ToQuboCallback) -> None:
        """Convert the `QuadraticProgram` into a QUBO using the provided callback function"""
        callback.set_num_variables(self.num_qubo_vars)

        def add_linear(var: VariableMappingInfo, coefficient: float) -> None:
            # Add a linear term to the QUBO by binary-mapping the integer (or, trivially,
            # binary) variable.
            if coefficient == 0.0:
                return
            for bit in range(var.bits):
                callback.add_linear(var.index + bit, coefficient * coeff(var, bit))
            # The variable's lowerbound generates a constant term
            callback.add_constant(coefficient * var.lowerbound)

        def add_quadratic(
            var1: VariableMappingInfo, var2: VariableMappingInfo, coefficient: float
        ) -> None:
            # Add a quadratic term to the QUBO by binary-mapping the integer (or, trivially,
            # binary) variable.
            if coefficient == 0.0:
                return
            for bit1 in range(var1.bits):
                qubo_index1 = var1.index + bit1
                for bit2 in range(var2.bits):
                    qubo_index2 = var2.index + bit2
                    qubo_bias = coefficient * coeff(var1, bit1) * coeff(var2, bit2)
                    if qubo_index1 > qubo_index2:  # move everything above the diagonal
                        callback.add_quadratic(qubo_index2, qubo_index1, qubo_bias)
                    elif qubo_index1 < qubo_index2:
                        callback.add_quadratic(qubo_index1, qubo_index2, qubo_bias)
                    else:  # qubo_index1 == qubo_index2, self-interaction a^2=a in a QUBO
                        callback.add_linear(qubo_index1, qubo_bias)

            # the variables' lower bounds generate linear cross-terms in the multiplication
            # as well as a constant lowerbound1*lowerbound2 term
            add_linear(var1, coefficient * var2.lowerbound)
            add_linear(var2, coefficient * var1.lowerbound)
            # the above double-counted var1.lowerbound * var2.lowerbound, compensate here
            callback.add_constant(-coefficient * var1.lowerbound * var2.lowerbound)

        def add_objective():
            # Add the `QuadraticProgram` objective terms to the QUBO
            for (
                _,
                var,
            ), coefficient in self.program.objective.linear.coefficients.items():
                add_linear(self.problem_variables[var], coefficient)

            for (
                var1,
                var2,
            ), coefficient in self.program.objective.quadratic.coefficients.items():
                add_quadratic(
                    self.problem_variables[var1],
                    self.problem_variables[var2],
                    coefficient,
                )

        def determine_objective_scale() -> float:
            # Retrieve the maximum scaling factor for the objective terms
            lin_b = self.program.objective.linear.bounds
            quad_b = self.program.objective.quadratic.bounds
            objective_scale = (
                1.0
                + (lin_b.upperbound - lin_b.lowerbound)
                + (quad_b.upperbound - quad_b.lowerbound)
            )
            logging.debug("objective has a scale of %f", objective_scale)
            return objective_scale

        def add_linear_equality_constraint(
            constraint: Constraint, lagrange: float
        ) -> None:
            # Add a linear equality constraint to the QUBO

            # we add (sum(a_i) - rhs)^2
            #   = sum(a_i^2) + rhs^2
            #     + 2*sum(a_i*a_j) - 2*rhs*sum(a_i)

            # sum(a_i^2) + 2*sum(a_i*a_j)  (implemented as sum (a_i*a_j) without enforcing j>i)
            for (_, var1), coefficient1 in constraint.linear.coefficients.items():
                for (_, var2), coefficient2 in constraint.linear.coefficients.items():
                    add_quadratic(
                        self.problem_variables[var1],
                        self.problem_variables[var2],
                        coefficient1 * coefficient2 * lagrange,
                    )

            # rhs^2
            callback.add_constant(constraint.rhs**2 * lagrange)

            # -2*rhs*sum(a_i)
            for (_, var), coefficient in constraint.linear.coefficients.items():
                add_linear(
                    self.problem_variables[var],
                    -2 * constraint.rhs * coefficient * lagrange,
                )

        def add_linear_inequality_constraint(
            c_index: int, constraint: Constraint, lagrange: float, sense: int
        ) -> None:
            # Add a linear inequality constraint to the QUBO.  There are some special inequality
            # constraint forms that do not require a slack variable but as they don't occur in our
            # models we're making no effort to detect those.
            assert self.slack_variables[c_index] is not None
            slack_variable = cast(VariableMappingInfo, self.slack_variables[c_index])

            # for <= we add (sum(a_i) - rhs + slack)^2
            #   = sum(a_i^2) + rhs^2 + slack^2
            #     + 2*sum(a_i*a_j) - 2*rhs*sum(a_i) + 2*slack sum(a_i) - 2*rhs*slack
            # for >= the sign for the slack variable flips

            # sum(a_i^2) + rhs^2 + 2*sum(a_i*a_j) - 2*rhs*sum(a_i)
            add_linear_equality_constraint(constraint, lagrange)

            # slack^2
            add_quadratic(slack_variable, slack_variable, lagrange)

            # 2*slack sum(a_i)
            for (_, var), coefficient in constraint.linear.coefficients.items():
                add_quadratic(
                    self.problem_variables[var],
                    slack_variable,
                    2 * sense * coefficient * lagrange,
                )

            # -2*rhs*slack
            add_linear(slack_variable, -2 * constraint.rhs * sense * lagrange)

        # Add the `QuadraticProgram` constraint terms to the QUBO
        def add_linear_constraints(lagrange: float):
            for c_index, constraint in enumerate(self.program.linear_constraints):
                if constraint.sense == ConstraintSense.EQ:
                    add_linear_equality_constraint(constraint, lagrange)
                elif constraint.sense == ConstraintSense.LE:
                    add_linear_inequality_constraint(c_index, constraint, lagrange, +1)
                elif constraint.sense == ConstraintSense.GE:
                    add_linear_inequality_constraint(c_index, constraint, lagrange, -1)
                else:
                    raise RuntimeError(
                        f"Constraint sense {constraint.sense} not implemented"
                    )

        # Quadratic constraints are not supported
        def add_quadratic_constraints(lagrange: float):  # pylint: disable=W0613
            if self.program.quadratic_constraints:
                raise NotImplementedError

        add_objective()
        lagrange = self.lagrange * determine_objective_scale()
        add_linear_constraints(lagrange)
        add_quadratic_constraints(lagrange)

    def interpret(self, qubo_result: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Interprets the QUBO result into a result for the original `QuadraticProgram` by
        reverse-mapping the integer variables and discarding the slack variables
        """
        result = np.empty([len(self.problem_variables)], dtype="int")
        for var, variable in enumerate(self.problem_variables):
            # np.empty() creates an uninitisalised array
            result[var] = variable.lowerbound

            for bit in range(variable.bits):
                result[var] += int(
                    qubo_result[variable.index + bit] * coeff(variable, bit)
                )

        return result

    def num_variables(self) -> int:
        """Return the number of variables in the mapped QUBO"""
        return self.num_qubo_vars


class EvaluateQuboCallback(ToQuboCallback):
    """Callback that evaluates a given QUBO result"""

    def __init__(
        self, result: Union[np.ndarray, List[float]], evaluate_constant: bool = False
    ) -> None:
        self._result = result
        self._evaluate_constant = evaluate_constant
        self.energy = 0.0
        super().__init__()

    def add_constant(self, constant: float) -> None:
        if self._evaluate_constant:
            self.energy += constant

    def add_quadratic(self, var1: int, var2: int, coefficient: float) -> None:
        self.energy += self._result[var1] * self._result[var2] * coefficient
