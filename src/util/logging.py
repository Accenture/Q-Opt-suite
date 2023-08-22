"""
Copyright (c) 2023 Objectivity Ltd.
"""

from dataclasses import dataclass
import datetime as dt
import logging
import os
from typing import Any, Iterable, Iterator, Optional, Tuple, Union
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
        self._platform_name = platform_name
        self._model_name = model_name
        self._size = size
        self._iteration = 0
        self._result = BenchmarkResult
        self._variables: Optional[int] = None
        self._timeout: Optional[int] = None
        self._is_error: bool = True
        self._is_feasible: bool = False
        self._quality: Optional[float] = None
        self._timings: Optional[list] = None
        self._solver_time: Optional[float] = None
        self._cost: Optional[float] = None
        self._rss: list[int] = []
        self._vms: list[int] = []
        self._info: str = ""

    def _avg_scalar(self, name: str, value: Optional[Union[int, float]]) -> None:
        """Update a simple average value"""
        if value is not None and (avg := getattr(self, name)) is not None:
            new_sum = value + avg * self._iteration
            setattr(self, name, new_sum / (self._iteration + 1))
        else:  # even a single None invalidates the entire average
            setattr(self, name, None)

    def _avg_list(self, name: str, values: Iterable[float]) -> None:
        """Update a list of average values"""
        average = getattr(self, name)
        for i, value in enumerate(values):
            new_sum = value + average[i] * self._iteration
            average[i] = new_sum / (self._iteration + 1)

    def _max_list(self, name: str, values: Iterable[float]) -> None:
        """Update a list of values keeping the maximum for each"""
        maximum = getattr(self, name)
        for i, value in enumerate(values):
            maximum[i] = max(maximum[i], value)

    def add_exception(self, info: Any):
        """:param info: the information to add"""
        self._info = f"{self._info}, {info}" if self._info else str(info)
        logging.exception(str(info))

    def add_result(
        self,
        result: BenchmarkResult,
    ) -> None:
        """
        Add the outcome of a single benchmark run to the result.

        :param result The benchmark result tuple
        """

        def time_diff(stat_pair: Tuple[Stat, Stat]) -> float:
            # calculate the time difference between two stat entries
            return round(stat_pair[1].time - stat_pair[0].time, 3)

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
        if not self._is_feasible:  # first (feasible) result
            self._variables = result.variables
            self._timeout = result.timeout
            self._is_error = False
            self._is_feasible = result.is_feasible
            self._quality = result.quality
            self._solver_time = result.solver_time
            self._cost = result.cost
            self._timings = list(timings)
            self._rss = list(rss)
            self._vms = list(vms)
            self._info = result.info
            self._iteration = 1
        elif result.is_feasible:  # subsequent results only incorporated if feasible
            assert self._variables == result.variables
            self._avg_scalar("_timeout", result.timeout)
            self._avg_scalar("_quality", result.quality)
            self._avg_scalar("_solver_time", result.solver_time)
            self._avg_scalar("_cost", result.cost)
            self._avg_list("_timings", timings)
            self._max_list("_rss", rss)
            self._max_list("_vms", vms)
            self._iteration += 1

    def to_csv(self) -> str:
        """
        :returns a CSV representation of the result
        """
        return (
            f'"{self._platform_name}",'
            f'"{self._model_name}",'
            f"{self._size},"
            f'{self._variables or ""},'
            f'{self._timeout or ""},'
            f"{self._is_error},"
            f"{self._is_feasible},"
            f'{self._quality if self._quality is not None else ""},'
            f'{",".join(map(str, self._timings)) if self._timings else ",,"},'
            f'{self._solver_time or ""},'
            f'{self._cost or ""},'
            f'{max(self._rss) if self._rss else ""},'
            f'{max(self._vms) if self._vms else ""},'
            f'"{self._info or ""}"'
        )


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

        with open(f"{self.directory}/config.yml", "w", encoding="utf-8") as out:
            yaml.dump(config, out)

        with open(f"{self.resultsfile}", "w", encoding="utf-8") as out:
            print(
                "Platform,Model,Size,Variables,Timeout,Error,Feasible,Quality,TTProblemInS,"
                "TTSolveInS,TTEvaluateInS,SolverTimeInS,Cost,RssMaxK,VmsMaxK,Info",
                file=out,
            )

    def log_summary(self, summary: BenchmarkSummary) -> None:
        """
        Log a single benchmark result to the results file.

        :param result: the benchmark result to log
        """
        with open(f"{self.resultsfile}", "a", encoding="utf-8") as out:
            print(summary.to_csv(), file=out)
