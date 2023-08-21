"""
Copyright (c) 2023 Objectivity Ltd.
"""

import logging
from typing import Any, cast
import dimod  # type: ignore
from dwave.system import LeapHybridSampler  # type: ignore
import numpy as np
from qiskit_optimization import QuadraticProgram  # type: ignore
from models.model import ModelStep

from platforms.dwaveleap import DWaveLEAP, DWaveQuboCallback


class DWaveBQM(DWaveLEAP):
    """
    The Platform class for the D-Wave Binary Quadratic Model sampler.
    """

    def __init__(self, config: dict) -> None:
        """
        Instantiate the D-Wave sampler with the given configuration taken from the yaml file.

        :param config: the model section of the configuration file
        """
        super().__init__(LeapHybridSampler, config)

    def translate_problem(self, step: ModelStep) -> dimod.BinaryQuadraticModel:
        """
        Translate the problem into a D-Wave `BinaryQuadraticModel`.

        :param model: the `ModelStep` to translate into the platform's native format
        :returns the model as a `BinaryQuadraticModel`
        """
        cb = DWaveQuboCallback()
        self.construct_qubo(step, cb)
        return cb.bqm

    def num_variables(self, problem: Any) -> int:
        """
        Determine the number of variabes in the `BinaryQuadraticModel`.

        :param problem: the `BinaryQuadraticModel` representing the model being optimised
        :returns the number of variables in the model
        """
        bqm = cast(dimod.BinaryQuadraticModel, problem)
        return bqm.num_variables

    def solve(
        self, problem: Any, timeout: int, num_solutions_desired: int
    ) -> dimod.SampleSet:
        """
        Solves the `BinaryQuadraticModel` and returns a `SampleSet`.

        :param problem: the problem being solved as a `BinaryQuadraticModel`
        :param timeout: the timeout in seconds
        :param num_solutions_desired: how many solutions should be returned
        :returns the optimised variable assignment
        """
        bqm = cast(dimod.BinaryQuadraticModel, problem)
        assert num_solutions_desired == 1  # TODO implement multiple result support

        label = f"Dwave-BQM-{bqm.num_variables}"
        sampleset = self.sampler.sample(bqm, label=label, time_limit=timeout)
        logging.info(f"DWave result energy={sampleset.first.energy}")
        return sampleset

    def translate_result(
        self,
        step: ModelStep,
        qubo: QuadraticProgram,
        result: Any,
        num_solutions_desired: int,
    ) -> np.ndarray:
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
        result = [sample[v] for v in range(step.converter.num_variables())]
        return step.from_qubo(result)
