"""
Copyright (c) 2023 Objectivity Ltd.
"""

import argparse
import logging
import random
import time
from typing import Iterator, List, Optional, cast
import importlib
import psutil
import yaml
from models.model import Model, ModelStep

from util.logging import BenchmarkLogger, BenchmarkSummary, BenchmarkResult, Stat
from platforms.platform import Platform


def create_platform_factory(
    platform_name: str, platform_config: dict, logger: BenchmarkLogger
) -> Iterator[Platform]:
    """
    Lazily instantiates a `Platform`. This improves execution time and decreases logfile spam.

    :param platform_name: the `Platform` class name
    :param platform_config: the relevant section from the configuration file
    :param logger: the `BenchmarkLogger` that may be used by the `Platform`
    :returns: an endless sequence of the instantiated `Platform` object.
    """
    class_name = platform_config.get("class", platform_name)
    module = importlib.import_module(f"platforms.{class_name.lower()}")
    platform = getattr(module, class_name)(platform_config)
    platform.logger = logger
    while True:
        yield platform


def single_benchmark(
    platform: Platform, model_class: type, config: dict
) -> BenchmarkResult:
    """
    Run a single benchmark for the given platform, model, and configuration

    :param platform: the `Platform` instance used to run the benchmark
    :param model_class: the `Model` type used to instantiate the problem being run
    :param config: the relevant section from the configuration file
    :returns the benchmark results
    """
    random.seed(0)  # make sure that the pseudo-RNG produces consistent results
    model: Model = model_class(config)  # instantiate the model
    timeout: int = int(config.get("timeout", 60))
    logging.info("Running size=%d timeout=%d", config["size"], timeout)

    # Translate the problem into the platform-specific format and do a timed run
    stats: list[Stat] = []

    def log_stats(info: str):
        memory_info = psutil.Process().memory_info()
        stat = Stat(time.time(), memory_info.rss, memory_info.vms)
        elapsed = stat.time - stats[len(stats) - 1].time if stats else 0
        logging.debug(
            "%s time=%ds rss=%dk vms=%dk", info, int(elapsed), stat.rss, stat.vms
        )
        stats.append(stat)

    info: List[Optional[str]] = []
    variables: List[int] = []
    solver_time: List[Optional[float]] = []
    cost: List[Optional[float]] = []

    def execute_step(step: ModelStep):
        platform_problem = platform.translate_problem(step)
        num_variables = platform.num_variables(platform_problem)
        log_stats(
            f"Problem formulated variables={num_variables}"
            + (
                ", {step.num_solutions_desired} solutions desired"
                if step.num_solutions_desired > 1
                else ""
            )
        )
        variables.append(num_variables)
        platform_result = platform.solve(
            platform_problem, timeout, step.num_solutions_desired
        )
        log_stats("Problem solved")
        info.append(platform.get_info(platform_result))
        solver_time.append(platform.get_solver_time(platform_result))
        cost.append(platform.get_cost(platform_result))
        step_result = platform.translate_result(
            step, step.program, platform_result, step.num_solutions_desired
        )
        log_stats("Result processed")
        return step_result

    log_stats("Initiating")
    result = model.execute(execute_step)

    # Interpret the result returned by the platform
    is_feasible = model.is_feasible(result)
    if is_feasible:
        quality = model.quality(result)
        logging.info("Completed with quality %f", quality)
    else:
        quality = None
        logging.warning("Completed but solution is infeasible")

    total_cost = sum(cast(List[float], cost)) if None not in cost else None
    total_time = (
        sum(cast(List[float], solver_time)) if None not in solver_time else None
    )
    total_info = ", ".join(filter(None, info))  # skip None values
    return BenchmarkResult(
        max(variables),
        timeout,
        is_feasible,
        quality,
        total_time,
        total_cost,
        stats,
        total_info,
    )


def model_benchmarks(  # pylint: disable=R0913
    platform_factory: Iterator[Platform],
    platform_name: str,
    model_name: str,
    sizes: list[dict],
    repeats: int,
    is_enabled: bool,
    logger: BenchmarkLogger,
) -> None:
    """
    Loop over the configured sizes for the given model and run the bencharks for them

    :param platform_factory: factory for the `Platform` used to run the benchmark
    :param platform_name: the name of the configured `Platform` being used
    :param model_name: the name of the `Model` being run
    :param sizes: the section of the configuration file specifying the model sizes and attributes
    :param repeats: how often the benchmark needs to be run and averaged
    :param is_enabled: whether the platform has been enabled in the configuration file,
        this may be overridden by the model size configuration
    :param logger: the `BenchmarkLogger` used to log the results
    """
    module = importlib.import_module(f"models.{model_name.lower()}")
    model_class = getattr(module, model_name)

    for size_config in sizes:
        if size_config.get("enabled", is_enabled):
            size = size_config["size"]
            result = BenchmarkSummary(platform_name, model_name, size)
            try:
                for _ in range(repeats):
                    result.add_result(
                        single_benchmark(
                            next(platform_factory), model_class, size_config
                        )
                    )
            except MemoryError:
                result.add_exception("Out of memory")
            except Exception as error:  # pylint: disable=W0718
                result.add_exception(error)
            finally:
                logger.log_summary(result)


def platform_benchmarks(
    platform_factory: Iterator[Platform],
    platform_name: str,
    models: dict[str, dict],
    repeats: int,
    is_enabled: bool,
    logger: BenchmarkLogger,
) -> None:
    """
    Loop over the configured models for the given platform and run the bencharks for them

    :param platform_factory: factory for the `Platform` used to run the benchmark
    :param platform_name: the name of the configured `Platform` being used
    :param models: the `Model` configuration for the given platform
    :param repeats: how often the benchmark needs to be run and averaged
    :param is_enabled: whether the platform has been enabled in the configuration file,
        this may be overridden by the model size configuration
    :param logger: the `BenchmarkLogger` used to log the results
    """
    for model_name, sizes in models.items():
        if sizes:
            model_benchmarks(
                platform_factory,
                platform_name,
                model_name,
                cast(list[dict], sizes),
                repeats,
                is_enabled,
                logger,
            )


class BenchmarkManager:  # pylint: disable=R0903
    """
    Manages the running of the configured benchmarks.
    """

    logger: BenchmarkLogger

    def __init__(self, config) -> None:
        """
        :param config: the benchmark configuration information (i.e. the yaml file supplied)
        """
        self.config = config
        self.logger = BenchmarkLogger(config)

    def benchmark(self) -> None:
        """
        Runs a complete suite of benchmarks, as specified in the configuration file
        that was passed to the constructor
        """
        for platform_name, platform_config in self.config["platforms"].items():
            # Loop over the configured hardware platforms
            is_enabled = bool(
                platform_config.get("enabled", self.config.get("enabled", True))
            )
            if (
                "models" in platform_config
            ):  # models have been specified with the platform
                models = platform_config["models"]
            elif (
                is_enabled
            ):  # using top level models... but only if the platform is enabled
                models = self.config.get("models")
            else:  # platform is disabled and no locally specified models that could override it
                logging.debug("Skipping %s because it is disabled", platform_name)
                continue

            if not models:
                logging.debug("Skipping %s because there are no models", platform_name)
                continue

            factory = create_platform_factory(
                platform_name, platform_config, self.logger
            )
            repeats = max(
                1, int(platform_config.get("repeats", self.config.get("repeats", 1)))
            )
            platform_benchmarks(
                factory, platform_name, models, repeats, is_enabled, self.logger
            )


def main() -> None:
    """Main function that triggers the benchmarks."""
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", help="Provide config file")
        args = parser.parse_args()
        config_name = args.config if args.config else "default.yml"
        logging.debug("Loading config file %s", config_name)
        config = yaml.load(open(config_name, encoding="utf-8"), Loader=yaml.FullLoader)
        BenchmarkManager(config).benchmark()

    except Exception as exception:
        logging.exception(exception)
        raise exception


if __name__ == "__main__":
    main()
