"""
Copyright (c) 2023 Objectivity Ltd.
"""

from io import StringIO
import logging
from math import sqrt
from typing import Dict, Iterable, Tuple

from qiskit_optimization import QuadraticProgram  # type: ignore
import yaml  # type: ignore

from models.model import SimpleModel

import models.pizzaparlour.generate_model as pm
import models.pizzaparlour.hamiltonian as ph


class _QiskitHamiltonianCallback(ph.HamiltonianCallback):
    """
    Callbacks used during model construction to build a representation the problem Hamiltonian.
    This implementation uses a `QuadraticProgram` to represent the problem.
    """

    def __init__(self, program: QuadraticProgram) -> None:
        """
        :param program: the `QuadraticProgram` to use to represent the problem Hamiltonian
        """
        super().__init__()
        self._program = program
        self._linear: Dict[str, float] = {}

    def add_linear(self, var: str, hi: int, cost: float) -> None:
        """
        Add a linear term to the problem Hamiltonian objective.

        :param var: the variable name
        :param hi: the variable upper bound
        :param cost: the cost coefficient
        """
        self._program.integer_var(name=var, upperbound=hi)
        self._linear[var] = cost

    def add_quadratic(
        self, var1: str, hi1: int, var2: str, hi2: int, cost: float
    ) -> None:
        """
        Add a quadratic term to the problem Hamiltonian.

        :raises RuntimeError: this class does not implement this callback
        """
        raise RuntimeError("Quadratic objective unimplemented")

    def set_objective(self) -> None:
        """
        Take the linear objective terms aggregated and add them to the `QuadraticProgram`.

        :param linear: the linear objective terms, defaults to everything set using `add_linear`
        """
        self._program.minimize(linear=self._linear)

    def add_constraint(
        self, name: str, tuples: Iterable[Tuple[str, int]], sense, value: float
    ) -> None:
        """
        Add a linear constraint to the problem Hamiltonian.

        :param name: the name of the constraint
        :param tuples: a list (or iterable) of `(pizza, ingredient_count)` tuples
        :param sense: the constraint sense <= == or >=
        :param value: the right hand side of the constraint
        """
        self._program.linear_constraint(
            linear={pizza: ingredient_count for (pizza, ingredient_count) in tuples},
            sense=sense,
            rhs=value,
            name=name,
        )


class Pizza(SimpleModel):
    """
    A model for the "pizza parlour" linear programming problem.

    On one hand, there is a range of ingredients available at given quantities. Each ingredient
    has a cost price.

    On the other hand, there is a range of pizza recipes we can make using the ingredients. Each
    pizza recipe has a given sales price.

    The optimisation goal is to maximise our profits while staying within the constraints of the
    given ingredient stock.

    This problem is not natural territory for annealers as it can be readily solved algorithmically,
    however, the purpose here is not to provide the most efficient implementation but rather to
    assess how solver platforms that do not natively support integers behave when confronted with
    the complex Hamiltonian landscape of binary-encoded integers.
    """

    def __init__(self, config: dict) -> None:
        """
        :param config: the model section of the configuration file
        """
        super().__init__(config)
        size = config["size"]
        ingredients = max(
            config.get("ingredients", int(sqrt(size))), 3
        )  # total number of ingredients
        avg_pizzas = config.get("avg_pizzas", 100)  # average pizzas per product line
        avg_min_pizzas = config.get(
            "avg_min_pizzas", 10
        )  # average minimum pizzas per product line
        avg_ingredients = config.get(
            "avg_ingredients", 5
        )  # average ingredients per pizza
        avg_price = config.get("avg_price", 20)  # average pizza price
        variability = config.get(
            "variability", 2
        )  # amount of variation around the means
        margin = config.get("margin", 2)  # the average profit margin

        self.model = pm.generate_model(
            num_ingredients=ingredients,
            num_pizzas=size,
            avg_pizzas=avg_pizzas,
            avg_min_pizzas=avg_min_pizzas,
            avg_ingredients=avg_ingredients,
            avg_price=avg_price,
            variability=variability,
            margin=margin,
        )
        with StringIO(f"Pizza model of size {size}:\n") as out:
            yaml.dump(self.model, out)
            logging.debug(out.getvalue())

        self.program = QuadraticProgram("Pizza")
        callback = _QiskitHamiltonianCallback(self.program)
        ph.pizza_profit(self.model, callback)
        callback.set_objective()
        ph.stock_constraints(self.model, callback)

    def quality(self, result) -> float:
        quality = super().quality(result)

        # paranoia check
        sample = {
            variable.name: result[i]
            for i, variable in enumerate(self.program.variables)
        }
        profit = ph.calculate_profit(self.model, sample)
        logging.debug("profit %d, total=%d", profit, sum(profit.values()))
        return quality

    def is_feasible(self, result) -> bool:
        is_feasible = super().is_feasible(result)

        # paranoia check
        sample = {
            variable.name: result[i]
            for i, variable in enumerate(self.program.variables)
        }
        return is_feasible and ph.is_valid(self.model, sample)
