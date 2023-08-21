"""
Copyright (c) 2023 Objectivity Ltd.
"""

from dataclasses import dataclass
import datetime as dt
from functools import reduce
import logging
import os
from typing import Iterable, Iterator, List, Optional, Tuple, Union
import yaml  # type: ignore


@dataclass
class Stat:
    """Timing and memory values."""

    time: float
    rss: int
    vms: int


@dataclass
class BenchmarkResult:
    """Benchmark result values."""

    variables: Optional[int]
    timeout: Optional[int]
    is_feasible: bool
    quality: Optional[float]
    solver_time: Optional[float]
    cost: Optional[float]
    stats: list[Stat]
    info: str


class BenchmarkSummary:
    """
    Simple container to encapsulate the results summary of running a single benchmark.
    """

    def __init__(self, platform_name: str, model_name: str, size: int) -> None:
        self.platform_name = platform_name
        self.model_name = model_name
        self.size = size
        self.iteration = 0
        self.variables: Optional[int] = None
        self.timeout: Optional[int] = None
        self.is_error: bool = True
        self.is_feasible: bool = False
        self.quality: Optional[float] = None
        self.timings: Optional[list] = None
        self.solver_time: Optional[float] = None
        self.cost: Optional[float] = None
        self.rss: list[int] = []
        self.vms: list[int] = []
        self.info: str = ""

    def _avg_scalar(self, name: str, value: Optional[Union[int, float]]) -> None:
        """Update a simple average value"""
        if value is not None and (avg := getattr(self, name)) is not None:
            new_sum = value + avg * self.iteration
            setattr(self, name, new_sum / (self.iteration + 1))
        else:  # even a single None invalidates the entire average
            setattr(self, name, None)

    def _avg_list(self, name: str, values: Iterable[float]) -> None:
        """Update a list of average values"""
        a = getattr(self, name)
        for i, value in enumerate(values):
            new_sum = value + a[i] * self.iteration
            a[i] = new_sum / (self.iteration + 1)

    def _max_list(self, name: str, values: Iterable[float]) -> None:
        """Update a list of values keeping the maximum for each"""
        a = getattr(self, name)
        for i, value in enumerate(values):
            a[i] = max(a[i], value)

    def add_result(
        self,
        result: BenchmarkResult,
    ) -> None:
        """
        Add the outcome of a single benchmark run to the result.

        :param result The benchmark result tuple
        """

        def time_diff(a: Tuple[Stat, Stat]) -> float:
            # calculate the time difference between two stat entries
            return round(a[1].time - a[0].time, 3)

        def create_groups(i: Iterator, group_size: int = 3):
            # iterate through the iterator in groups (of size 3 by default)
            while True:
                try:
                    yield [next(i) for _ in range(group_size)]
                except StopIteration:
                    break

        timings = map(
            sum,  # each step produces 3 timings, add up each of the 3 timings across the steps
            zip(*create_groups(map(time_diff, zip(result.stats, result.stats[1:])))),
        )
        rss = map(lambda s: s.rss // 1024, result.stats)
        vms = map(lambda s: s.vms // 1024, result.stats)
        if not self.is_feasible:  # first (feasible) result
            self.variables = result.variables
            self.timeout = result.timeout
            self.is_error = False
            self.is_feasible = result.is_feasible
            self.quality = result.quality
            self.solver_time = result.solver_time
            self.cost = result.cost
            self.timings = list(timings)
            self.rss = list(rss)
            self.vms = list(vms)
            self.info = result.info
            self.iteration = 1
        elif result.is_feasible:  # subsequent results only incorporated if feasible
            assert self.variables == result.variables
            self._avg_scalar("timeout", result.timeout)
            self._avg_scalar("quality", result.quality)
            self._avg_scalar("solver_time", result.solver_time)
            self._avg_scalar("cost", result.cost)
            self._avg_list("timings", timings)
            self._max_list("rss", rss)
            self._max_list("vms", vms)
            self.iteration += 1


class BenchmarkLogger:
    """
    Handles the logging of benchmark results.
    """

    LOGGING_LEVELS = {
        "critical": logging.FATAL,
        "fatal": logging.FATAL,
        "error": logging.ERROR,
        "warning": logging.WARN,
        "warn": logging.WARN,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    def __init__(self, config: dict) -> None:
        """
        Configure Python logging and create the benchmark log and results files.
        """
        self.directory = f"results/{dt.datetime.now():%Y-%m-%d-%H.%M.%S}"
        self.resultsfile = f"{self.directory}/results.csv"
        os.makedirs(self.directory)

        level = config.get("logging", "info").lower()
        if level not in BenchmarkLogger.LOGGING_LEVELS:
            raise ValueError(f"Unknown logging level {level}")

        logging.root.handlers = []
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"{self.directory}/logfile.log"),
                logging.StreamHandler(),
            ],
            level=BenchmarkLogger.LOGGING_LEVELS[level],
            force=True,
        )

        with open(f"{self.directory}/config.yml", "w") as out:
            yaml.dump(config, out)

        with open(f"{self.resultsfile}", "w") as out:
            print(
                f"Platform,Model,Size,Variables,Timeout,Error,Feasible,Quality,TTProblemInS,TTSolveInS,TTEvaluateInS,SolverTimeInS,Cost,RssMaxK,VmsMaxK,Info",
                file=out,
            )

    def log_result(self, result: BenchmarkSummary) -> None:
        """
        Log a single benchmark result to the results file.

        :param result: the benchmark result to log
        """
        with open(f"{self.resultsfile}", "a") as out:
            print(
                f'"{result.platform_name}",'
                f'"{result.model_name}",'
                f"{result.size},"
                f'{result.variables or ""},'
                f'{result.timeout or ""},'
                f"{result.is_error},"
                f"{result.is_feasible},"
                f'{result.quality if result.quality is not None else ""},'
                f'{",".join(map(str, result.timings)) if result.timings else ",,"},'
                f'{result.solver_time or ""},'
                f'{result.cost or ""},'
                f'{max(result.rss) if result.rss else ""},'
                f'{max(result.vms) if result.vms else ""},'
                f'"{result.info or ""}"',
                file=out,
            )
