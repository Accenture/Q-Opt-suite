"""
Copyright (c) 2023 Objectivity Ltd.
"""

import logging
from typing import Any, cast
import dimod  # type: ignore
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite  # type: ignore
import numpy as np
from qiskit_optimization import QuadraticProgram  # type: ignore
from models.model import ModelStep

from platforms.dwaveleap import DWaveQuboCallback, DWaveLEAP


class DWaveQPU(DWaveLEAP):
    """
    The Platform class for the D-Wave QPU sampler.
    """

    def __init__(self, config: dict) -> None:
        """
        Instantiate the D-Wave sampler with the given configuration taken from the yaml file.

        :param config: the model section of the configuration file
        """
        super().__init__(DWaveSampler, config)
        self.num_reads: int = config.get("num_reads", 1000)
        self.embedded: FixedEmbeddingComposite = None

    def translate_problem(self, step: ModelStep) -> dimod.BinaryQuadraticModel:
        """
        Translate the problem into a D-Wave `BinaryQuadraticModel`. Also fixes the
        embedding as a side effect.

        :param model: the `ModelStep` to translate
        :returns the model as a `BinaryQuadraticModel`
        """
        callback = DWaveQuboCallback()
        self.construct_qubo(step, callback)

        sampleset = EmbeddingComposite(self.sampler).sample(
            callback.bqm,
            return_embedding=True,
            answer_mode="raw",
            num_reads=1,
            annealing_time=1,
        )
        embedding = sampleset.info["embedding_context"]["embedding"]
        self.embedded = FixedEmbeddingComposite(self.sampler, embedding)
        return callback.bqm

    def num_variables(self, problem: Any) -> int:
        """
        Determines the number of variabes in the `BinaryQuadraticModel`.

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

        sampleset = self.embedded.sample(
            bqm, answer_mode="raw", num_reads=self.num_reads, annealing_time=timeout
        )
        logging.info("DWave result energy=%f", sampleset.first.energy)
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
        :param result: the optimised variable assignment of as a `SampleSet`
        :param num_solutions_desired: how many solutions should be returned
        :returns the optimised variable assignment(s) in terms of the `QuadraticProgram`.
        """
        assert num_solutions_desired == 1  # TODO implement multiple result support
        sampleset = cast(dimod.SampleSet, result)

        sample = sampleset.first.sample
        result = [sample[v] for v in range(step.converter.num_variables())]
        return step.from_qubo(result)
