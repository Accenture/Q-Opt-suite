"""
Copyright (c) 2023 Objectivity Ltd.
"""

from abc import abstractmethod
from collections import namedtuple
from itertools import repeat
import logging
from typing import List, Optional, Union, cast
from math import log2, ceil
from array import array
import numpy as np

from qiskit_optimization import QuadraticProgram # type: ignore
from qiskit_optimization.problems.variable import Variable # type: ignore
from qiskit_optimization.problems.constraint import Constraint, ConstraintSense # type: ignore


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
        pass

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
        logging.debug(f"offset={self.constant}")
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
    def convert(self, cb: ToQuboCallback) -> None:
        """
        Convert the encapsulated QuadraticProgram to a QUBO using the callback.
        """
        raise NotImplementedError

    @abstractmethod
    def interpret(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
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

    def convert(self, cb: ToQuboCallback) -> None:
        """
        Constructs a platform-specific QUBO directly from the information in the
        `QuadraticProgram` as the latter is already a QUBO
        """
        cb.set_num_variables(len(self.program.variables))

        cb.add_constant(self.program.objective.constant)

        for (_, x), coefficient in self.program.objective.linear.coefficients.items():
            cb.add_linear(x, coefficient)

        for (
            x,
            y,
        ), coefficient in self.program.objective.quadratic.coefficients.items():
            if x != y:
                cb.add_quadratic(x, y, coefficient)
            else:  # the Qiskit QuadraticProgramToQubo converter commits this horror
                cb.add_linear(x, coefficient)

    def interpret(self, result: Union[np.ndarray, list[float]]) -> np.ndarray:
        """Returns `result`. The QUBO results need no interpretation in this identity mapping"""
        return cast(np.ndarray, result)

    def num_variables(self) -> int:
        """Returns the number of variables in the QUBO, identical to the `QuadraticProgram`"""
        return len(self.program.variables)


"""
Structured tuple storing information about the mapping of a problem variable to QUBO

- `index` the index of the variable in the QUBO model
- `bits` the number of bits occupied in the QUBO model
- `msb_coef` the coefficient of the most significant bit (the others are `2**bit_position`)
- `upperbound` the upper bound (inclusive) of the variable
- `lowerbound` the lower bound (inclusive) of the variable
"""
VariableMappingInfo = namedtuple(
    "VariableMappingInfo", ["index", "bits", "msb_coef", "upperbound", "lowerbound"]
)


def coeff(var: VariableMappingInfo, b: int) -> float:
    """
    Return the value of the given bit in the variable. This is `2**bit_position` except
    for the most significant bit which has a value to match the range of the variable.
    """
    return float(2**b) if b < var.bits - 1 else var.msb_coef


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
            # The coefficient of the most significant bit is tuned to produce exactly the variable range
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
                    f"adding binary problem variable {variable.name} at position {var.index}"
                )
            elif variable.vartype == Variable.Type.INTEGER:
                var = allocate_variable(variable.upperbound, variable.lowerbound)
                logging.debug(
                    f"adding {var.bits} bit integer problem variable {variable.name} at position {var.index}"
                )
            else:
                raise RuntimeError(f"Variable type {variable.vartype} not implemented")
            return var

        def allocate_slack_variable(constraint: Constraint) -> Optional[VariableMappingInfo]:
            # Allocate QUBO variable space for the given slack variable. There are some special inequality
            # constraint forms that do not require a slack variable but as they don't occur in our models
            # we're making no effort to detect those.
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
                f"adding slack variable of {var.bits} bits for {constraint.name} at position {var.index}"
            )
            return var

        # Create the QUBO variable mapping for the `QuadraticProgram`
        self.problem_variables = list(map(allocate_problem_variable, program.variables))
        self.slack_variables = list(map(allocate_slack_variable, program.linear_constraints))
        if program.quadratic_constraints:
            raise RuntimeError("Quadratic constraints not implemented")

    def convert(self, cb: ToQuboCallback) -> None:
        """Convert the `QuadraticProgram` into a QUBO using the provided callback function"""
        cb.set_num_variables(self.num_qubo_vars)

        def add_linear(var: VariableMappingInfo, coefficient: float) -> None:
            # Add a linear term to the QUBO by binary-mapping the integer (or, trivially, binary) variable.
            if coefficient == 0.0:
                return
            for b in range(var.bits):
                cb.add_linear(var.index + b, coefficient * coeff(var, b))
            # The variable's lowerbound generates a constant term
            cb.add_constant(coefficient * var.lowerbound)

        def add_quadratic(
            var1: VariableMappingInfo, var2: VariableMappingInfo, coefficient: float
        ) -> None:
            # Add a quadratic term to the QUBO by binary-mapping the integer (or, trivially, binary) variable.
            if coefficient == 0.0:
                return
            for b1 in range(var1.bits):
                qubo_index1 = var1.index + b1
                for b2 in range(var2.bits):
                    qubo_index2 = var2.index + b2
                    qubo_bias = coefficient * coeff(var1, b1) * coeff(var2, b2)
                    if qubo_index1 > qubo_index2:  # move everything above the diagonal
                        cb.add_quadratic(qubo_index2, qubo_index1, qubo_bias)
                    elif qubo_index1 < qubo_index2:
                        cb.add_quadratic(qubo_index1, qubo_index2, qubo_bias)
                    else:  # qubo_index1 == qubo_index2, self-interaction a^2=a in a QUBO
                        cb.add_linear(qubo_index1, qubo_bias)

            # the variables' lower bounds generate linear cross-terms in the multiplication
            # as well as a constant lowerbound1*lowerbound2 term
            add_linear(var1, coefficient * var2.lowerbound)
            add_linear(var2, coefficient * var1.lowerbound)
            # the above double-counted var1.lowerbound * var2.lowerbound, compensate here
            cb.add_constant(-coefficient * var1.lowerbound * var2.lowerbound)

        def add_objective():
            # Add the `QuadraticProgram` objective terms to the QUBO
            for (
                _,
                v,
            ), coefficient in self.program.objective.linear.coefficients.items():
                add_linear(self.problem_variables[v], coefficient)

            for (
                v1,
                v2,
            ), coefficient in self.program.objective.quadratic.coefficients.items():
                add_quadratic(
                    self.problem_variables[v1], self.problem_variables[v2], coefficient
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
            logging.debug(f"objective has a scale of {objective_scale}")
            return objective_scale

        def add_linear_equality_constraint(
            constraint: Constraint, lagrange: float
        ) -> None:
            # Add a linear equality constraint to the QUBO

            # we add (sum(a_i) - rhs)^2
            #   = sum(a_i^2) + rhs^2
            #     + 2*sum(a_i*a_j) - 2*rhs*sum(a_i)

            # sum(a_i^2) + 2*sum(a_i*a_j)  (implemented as sum (a_i*a_j) without enforcing j>i)
            for (_, v1), coefficient1 in constraint.linear.coefficients.items():
                for (_, v2), coefficient2 in constraint.linear.coefficients.items():
                    add_quadratic(
                        self.problem_variables[v1],
                        self.problem_variables[v2],
                        coefficient1 * coefficient2 * lagrange,
                    )

            # rhs^2
            cb.add_constant(constraint.rhs**2 * lagrange)

            # -2*rhs*sum(a_i)
            for (_, v), coefficient in constraint.linear.coefficients.items():
                add_linear(
                    self.problem_variables[v],
                    -2 * constraint.rhs * coefficient * lagrange,
                )

        def add_linear_inequality_constraint(
            c: int, constraint: Constraint, lagrange: float, sense: int
        ) -> None:
            # Add a linear inequality constraint to the QUBO.  There are some special inequality
            # constraint forms that do not require a slack variable but as they don't occur in our models
            # we're making no effort to detect those.
            assert self.slack_variables[c] is not None
            slack_variable = cast(VariableMappingInfo, self.slack_variables[c])

            # for <= we add (sum(a_i) - rhs + slack)^2
            #   = sum(a_i^2) + rhs^2 + slack^2
            #     + 2*sum(a_i*a_j) - 2*rhs*sum(a_i) + 2*slack sum(a_i) - 2*rhs*slack
            # for >= the sign for the slack variable flips

            # sum(a_i^2) + rhs^2 + 2*sum(a_i*a_j) - 2*rhs*sum(a_i)
            add_linear_equality_constraint(constraint, lagrange)

            # slack^2
            add_quadratic(slack_variable, slack_variable, lagrange)

            # 2*slack sum(a_i)
            for (_, v), coefficient in constraint.linear.coefficients.items():
                add_quadratic(
                    self.problem_variables[v],
                    slack_variable,
                    2 * sense * coefficient * lagrange,
                )

            # -2*rhs*slack
            add_linear(slack_variable, -2 * constraint.rhs * sense * lagrange)

        # Add the `QuadraticProgram` constraint terms to the QUBO
        def add_linear_constraints(lagrange: float):
            for c, constraint in enumerate(self.program.linear_constraints):
                if constraint.sense == ConstraintSense.EQ:
                    add_linear_equality_constraint(constraint, lagrange)
                elif constraint.sense == ConstraintSense.LE:
                    add_linear_inequality_constraint(c, constraint, lagrange, +1)
                elif constraint.sense == ConstraintSense.GE:
                    add_linear_inequality_constraint(c, constraint, lagrange, -1)
                else:
                    raise RuntimeError(
                        f"Constraint sense {constraint.sense} not implemented"
                    )

        # Quadratic constraints are not supported
        def add_quadratic_constraints(lagrange: float):
            if self.program.quadratic_constraints:
                raise NotImplementedError

        add_objective()
        lagrange = self.lagrange * determine_objective_scale()
        add_linear_constraints(lagrange)
        add_quadratic_constraints(lagrange)

    def interpret(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Interprets the QUBO result into a result for the original `QuadraticProgram` by
        reverse-mapping the integer variables and discarding the slack variables
        """
        result = np.empty([len(self.problem_variables)], dtype="int")
        for v, var in enumerate(self.problem_variables):
            result[v] = var.lowerbound  # np.empty() creates an uninitisalised array
            for b in range(var.bits):
                result[v] += int(x[var.index + b] * coeff(var, b))

        return result

    def num_variables(self) -> int:
        """Return the number of variables in the mapped QUBO"""
        return self.num_qubo_vars


class EvaluateQuboCallback(ToQuboCallback):
    """Callback that evaluates a given QUBO result"""

    def __init__(
        self, x: Union[np.ndarray, List[float]], evaluate_constant: bool = False
    ) -> None:
        self._x = x
        self._evaluate_constant = evaluate_constant
        self.energy = 0.0
        super().__init__()

    def add_constant(self, constant: float) -> None:
        if self._evaluate_constant:
            self.energy += constant

    def add_quadratic(self, var1: int, var2: int, coefficient: float) -> None:
        self.energy += self._x[var1] * self._x[var2] * coefficient
