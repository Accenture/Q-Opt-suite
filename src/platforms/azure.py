"""
Copyright (c) 2023 Objectivity Ltd.
"""

from typing import Any, List, cast
from azure.quantum import Workspace  # type: ignore
import azure.quantum.optimization as aqo  # type: ignore
from qiskit_optimization import QuadraticProgram  # type: ignore

from models.model import ModelStep
from models.qubo import QuadraticProgramConverter, ToQuboDeduplicatingCallback
from platforms.platform import Platform


class AzureQuboCallback(ToQuboDeduplicatingCallback):
    """
    QUBO converter callback function which populates an Azure `Problem`
    """

    def __init__(self, name: str, is_lower_half: bool = True) -> None:
        super().__init__(is_lower_half)
        self._qubo = aqo.Problem(name=name, problem_type=aqo.ProblemType.pubo)

    # Type conversions because the MS library doesn't like the non-serializable
    # numpy data types that Qiskit uses

    def set_constant(self, constant: float) -> None:
        self._qubo.add_term(c=float(constant), indices=[])

    def set_linear(self, var: int, coefficient: float) -> None:
        self._qubo.add_term(c=float(coefficient), indices=[int(var)])

    def set_quadratic(self, var1: int, var2: int, coefficient: float) -> None:
        self._qubo.add_term(c=float(coefficient), indices=[int(var1), int(var2)])

    def get_qubo(self):
        self.set_interactions()
        return self._qubo


class Azure(Platform):
    """
    The `Platform` implementation for Azure.

    WARNING: Microsoft has deprecated all annealer technology on Azure.
    """

    def __init__(self, config: dict) -> None:
        """
        Instantiate the Azure `Workspace` with the given configuration taken from the yaml file.

        :param config: the model section of the configuration file
        """
        super().__init__(config)

        location = config["location"] if "location" in config else "uksouth"

        if "resource_id" in config:
            self._workspace = Workspace(
                resource_id=config["resource_id"], location=location
            )
        elif (
            "subscription_id" in config
            and "resource_group" in config
            and "name" in config
        ):
            self._workspace = Workspace(
                subscription_id=config["subscription_id"],
                resource_group=config["resource_group"],
                name=config["name"],
                location=location,
            )
        else:
            raise ValueError(
                "Azure: resource ID or subscription/group/name not specified in config"
            )

    def translate_problem(self, step: ModelStep) -> aqo.Problem:
        """
        Translate the problem into an Azure `Problem`

        :param step: the model step representing the problem
        :returns the equivalent Azure `Problem`
        """
        cb = AzureQuboCallback(f"{self}-{step}")
        self.construct_qubo(step, cb)
        return cb.get_qubo()

    def num_variables(self, problem: aqo.Problem) -> int:
        """
        Returns the number of variabes in the `Problem`

        :param problem: the Azure `Problem`
        :returns the number of variables
        """
        set_vars = set()

        for term in problem.terms:
            set_vars.update(term.ids)

        for term in problem.terms_slc:
            for subterm in term.terms:
                set_vars.update(subterm.ids)

        return len(set_vars)

    def translate_result(
        self,
        step: ModelStep,
        qubo: QuadraticProgram,
        result: Any,
        num_solutions_desired: int,
    ) -> list:
        """
        Translates the `Problem` result into variable values for the `QuadraticProgram`

        :param step: the model's optimisation step just executed
        :param qubo: the original QUBO as a `QuadraticProgram`
        :param result: the optimised variable assignment(s) in terms of the platform-specific model
        :param num_solutions_desired: the number of (near-)optimal solutions desired
        :returns the optmised variable assignment(s) in terms of the `QuadraticProgram`
        """
        results: List[Any] = []
        for i in range(min(num_solutions_desired, len(result["solutions"]))):
            configuration = result["solutions"][i]["configuration"]
            result = [
                int(configuration[str(v)])
                for v in range(step.converter.num_variables())
            ]
            results.append(step.from_qubo(result))

        return results[0] if num_solutions_desired == 1 else results
