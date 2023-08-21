"""
Copyright (c) 2023 Objectivity Ltd.
"""

from typing import Any, Optional, cast
import numpy as np
from qiskit_optimization import QuadraticProgram # type: ignore

import quantagonia.qubo as qq # type: ignore
from quantagonia.enums import HybridSolverConnectionType, HybridSolverOptSenses # type: ignore
from quantagonia.runner_factory import RunnerFactory # type: ignore
from quantagonia.spec_builder import QUBOSpecBuilder # type: ignore

from models.model import ModelStep
from models.qubo import ToQuboCallback
from platforms.platform import Platform


class QuantagoniaQuboCallback(ToQuboCallback):
    """
    Converter callback function which populates a Quantagonia `QuboModel`.
    """

    def __init__(self) -> None:
        self.qubo: qq.QuboModel = qq.QuboModel(sense=HybridSolverOptSenses.MINIMIZE)

    def set_num_variables(self, num_variables: int) -> None:
        self.vars: list = [
            self.qubo.addVariable(name=str(v)) for v in range(num_variables)
        ]

    def add_constant(self, constant: float) -> None:
        self.qubo.objective += constant

    def add_linear(self, var: int, coefficient: float) -> None:
        self.qubo.objective += coefficient * self.vars[var]

    def add_quadratic(self, var1: int, var2: int, coefficient: float) -> None:
        self.qubo.objective += coefficient * self.vars[var1] * self.vars[var2]


class Quantagonia(Platform):
    """
    The `Platform` implementation for Quantagonia.
    """

    def __init__(self, config: dict) -> None:
        """
        Instantiate the Quantagonia `Runner` with the given configuration taken from the yaml file.

        :param config: the model section of the configuration file
        """
        super().__init__(config)

        # As the sampler is always the same we can instantiate it here
        if "api_key" not in config:
            raise ValueError("Quantagonia API key not specified in config")
        self._runner: RunnerFactory = RunnerFactory.getRunner(
            HybridSolverConnectionType.CLOUD, api_key=config["api_key"]
        )
        self._presolve: Optional[bool] = config.get("presolve", None)

    def translate_problem(self, step: ModelStep) -> qq.QuboModel:
        """
        Translate the problem into a Quantagonia `QuboModel`.

        :param model: the `ModelStep` to translate
        :returns the model as a Quantagonia `QuboModel`
        """
        cb = QuantagoniaQuboCallback()
        self.construct_qubo(step, cb)
        return cb.qubo

    def num_variables(self, problem: Any) -> int:
        """
        :param problem: the Quantagonia `QuboModel`
        :returns the number of variabes in the `QuboModel`
        """
        quboModel = cast(qq.QuboModel, problem)
        return len(quboModel.vars)

    def solve(
        self, problem: Any, timeout: int, num_solutions_desired: int
    ) -> qq.QuboModel:
        """
        Solves the `QuboModel` and returns it as Quantagonia embeds the results in the model.

        :param problem: the problem being solved as a `QuboModel`
        :param timeout: the timeout in seconds
        :param num_solutions_desired: how many solutions should be returned
        :returns the `QuboModel` now containing the optimised variable assignment
        """
        assert num_solutions_desired == 1  # TODO implement multiple result support
        quboModel = cast(qq.QuboModel, problem)

        spec = QUBOSpecBuilder()
        if self._presolve is not None:
            spec.set_presolve(self._presolve)
        if timeout:
            spec.set_time_limit(timeout)
        quboModel.solve(spec.getd(), runner=self._runner)
        return quboModel

    def translate_result(
        self,
        step: ModelStep,
        qubo: QuadraticProgram,
        result: Any,
        num_solutions_desired: int,
    ) -> np.ndarray:
        """
        Translates the `QuboModel` result into variable values for the `QuadraticProgram`.
        """
        assert num_solutions_desired == 1  # TODO implement multiple result support
        quboModel = cast(qq.QuboModel, result)

        result = [
            quboModel.vars[str(v)].eval() for v in range(step.converter.num_variables())
        ]
        return step.from_qubo(result)
