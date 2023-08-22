"""
Copyright (c) 2023 Objectivity Ltd.
"""

import functools as ft
import operator as op

# callback abstracting the QCBO/QUBO model
from abc import abstractmethod


class HamiltonianCallback:
    """Callback function used to construct the Hamiltonian."""

    @abstractmethod
    def add_linear(self, var, hi, cost):  # pylint: disable=C0103
        """Add a linear term to the Hamiltonian."""

    @abstractmethod
    def add_quadratic(self, var1, hi1, var2, hi2, cost):  # pylint: disable=R0913
        """Add a quadratic term to the Hamiltonian."""

    @abstractmethod
    def set_objective(self):
        """Add an objective term to the Hamiltonian."""

    @abstractmethod
    def add_constraint(self, name, tuples, sense, value):
        """Add a constraint to the Hamiltonian."""


def _get_cost_ceiling(model, ingredient, quantity):
    stock, price = op.itemgetter("stock", "price")(model["inventory"][ingredient])
    return (price * quantity, stock // quantity)


def _summarise_cost_ceiling(ingredient1, ingredient2):
    cost1, ceil1 = ingredient1
    cost2, ceil2 = ingredient2
    return (cost1 + cost2, min(ceil1, ceil2))


def pizza_profit(model, callback: HamiltonianCallback):
    """calculate profit and roll it into the hamiltonian"""
    for name, properties in model["pizzas"].items():
        price, floor, ingredients = op.itemgetter(
            "price", "minimum_quantity", "ingredients"
        )(properties)
        cost, ceiling = ft.reduce(
            _summarise_cost_ceiling,
            map(lambda t: _get_cost_ceiling(model, *t), ingredients.items()),
        )
        callback.add_linear(name, ceiling - floor, cost - price)


# minimum quantity constraints
# these are not modelled explicitly, we're optimising pizzas _on top of_ the minimum quantity
# this means that we need to calculate effective stock levels


def stock_constraints(model, callback: HamiltonianCallback):
    """ingredient stock constraints"""
    inventory, pizzas = op.itemgetter("inventory", "pizzas")(model)

    def _reduce_by_minimum_pizzas(stock, pizza_properties):
        minimum_pizzas, pizza_ingredients = op.itemgetter(
            "minimum_quantity", "ingredients"
        )(pizza_properties)
        for ingredient_name, ingredient_quantity in pizza_ingredients.items():
            stock[ingredient_name] -= minimum_pizzas * ingredient_quantity
        return stock

    def _get_pizzas_for_ingredient(ingredient_name, constraints, pizza_data):
        pizza_name, pizza_properties = pizza_data
        ingredients = pizza_properties["ingredients"]
        if ingredient_name in ingredients:
            constraints.append((pizza_name, ingredients[ingredient_name]))
        return constraints

    effective_stock = ft.reduce(
        _reduce_by_minimum_pizzas,
        pizzas.values(),
        dict(
            map(
                lambda name, properties: (name, properties["stock"]),
                inventory.keys(),
                inventory.values(),
            )
        ),
    )

    for ingredient_name in inventory.keys():
        qqq = ft.partial(_get_pizzas_for_ingredient, ingredient_name)
        callback.add_constraint(
            f"{ingredient_name}-limit",
            ft.reduce(qqq, pizzas.items(), []),
            "<=",
            effective_stock[ingredient_name],
        )


def sample_to_production(model, sample):
    """calculate actual numbers on a given production run by adding minimum quantities back in"""
    pizzas = model["pizzas"]
    return dict(
        map(
            lambda pizza, quantity: (
                pizza,
                quantity + pizzas[pizza]["minimum_quantity"],
            ),
            sample.keys(),
            sample.values(),
        )
    )


def calculate_profit(model, sample):
    """calculate profit on a given production run"""

    def _pizza_profit(name, quantity):
        price, floor, ingredients = op.itemgetter(
            "price", "minimum_quantity", "ingredients"
        )(model["pizzas"][name])
        cost, ceiling = ft.reduce(
            _summarise_cost_ceiling,
            map(lambda t: _get_cost_ceiling(model, *t), ingredients.items()),
        )
        if quantity + floor > ceiling:
            print(
                f"WARNING {name} quantity {quantity + floor} > {ceiling} constraint breach"
            )
        return (name, (price - cost) * (quantity + floor))

    return dict(map(_pizza_profit, sample.keys(), sample.values()))


def check_inventory(model, sample):
    """constraint check on inventory"""
    inventory, pizzas = op.itemgetter("inventory", "pizzas")(model)

    def _consume_stock(stock, name, quantity):
        for ingredient, amount in pizzas[name]["ingredients"].items():
            stock[ingredient] -= amount * (quantity + pizzas[name]["minimum_quantity"])
        return stock

    stock = dict(
        map(
            lambda name, properties: (name, properties["stock"]),
            inventory.keys(),
            inventory.values(),
        )
    )

    return ft.reduce(lambda s, t: _consume_stock(s, *t), sample.items(), stock)


def is_valid(model, sample):
    """:returns whether the sample represents a valid solution"""
    try:
        next(
            filter(
                lambda remaining_stock: remaining_stock < 0,
                check_inventory(model, sample).values(),
            )
        )
    except StopIteration:
        return True
    else:
        return False
