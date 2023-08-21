"""
Copyright (c) 2023 Objectivity Ltd.
"""

from typing import Any, cast
import dimod  # type: ignore
from models.qubo import ToQuboCallback

from platforms.platform import Platform


class DWaveQuboCallback(ToQuboCallback):
    """
    Converter callback function which populates a DWave `dimod.BinaryQuadraticModel`
    """

    def __init__(self) -> None:
        pass

    def set_num_variables(self, num_variables: int) -> None:
        self.bqm = dimod.BinaryQuadraticModel(num_variables, dimod.BINARY)

    def add_constant(self, constant: float) -> None:
        self.bqm.add_offset(constant)

    def add_linear(self, var: int, coefficient: float) -> None:
        self.bqm.add_linear(var, coefficient)

    def add_quadratic(self, var1: int, var2: int, coefficient: float) -> None:
        self.bqm.add_quadratic(var1, var2, coefficient)


class DWaveLEAP(Platform):
    """
    The Platform class for the D-Wave LEAP cloud service.
    """

    def __init__(self, sampler, config: dict) -> None:
        """
        Instantiate the D-Wave sampler of the given class with the configuration taken
        from the yaml file

        :param config: the model section of the configuration file
        """
        super().__init__(config)

        if "token" not in config:
            raise ValueError("DWaveLEAP token not specified in config")
        self.sampler = sampler(token=config["token"])

    def get_solver_time(self, result: Any) -> float:
        """
        Find the actual solver time used in seconds.

        @param result: the optimised `SampleSet`
        @returns the charged solver time in seconds
        """
        # interesting are run_time, charge_time, qpu_access_time
        # See https://docs.dwavesys.com/docs/latest/doc_leap_hybrid.html
        samples = cast(dimod.SampleSet, result)
        timing = samples.info.get(
            "timing", samples.info
        )  # doesn't seem to be consistly the same?
        return float(timing.get("charge_time", timing.get("qpu_access_time"))) / 1e6
