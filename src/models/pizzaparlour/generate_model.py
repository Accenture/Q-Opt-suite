"""
Copyright (c) 2023 Objectivity Ltd.
"""

import random as r


def generate_model(
    num_ingredients: int,
    num_pizzas: int,
    avg_pizzas: int = 100,
    avg_min_pizzas: int = 10,
    avg_ingredients: int = 5,
    avg_price: float = 20,
    variability: int = 2,
    margin: float = 2,
) -> dict:
    """
    Generate a synthetic model of given size.

    :param ingredients:   total number of ingredients
    :param avg_pizzas:    average pizzas per product line
    :param avg_ingredients: average ingredients per pizza
    :param avg_price:     average pizza price
    :param variability:   amount of variation around the means
    :param margin:        the average profit margin
    :returns the model
    """
    def vary(n):
        return int(n * variability ** (r.random() * 2 - 1))

    avg_ingredient_stock = num_pizzas * avg_pizzas * avg_ingredients // num_ingredients
    avg_ingredient_price = avg_price // avg_ingredients // margin

    model: dict = {
        "inventory": {
            "dough": {"stock": num_pizzas * avg_pizzas, "price": avg_ingredient_price},
            "tomato": {
                "stock": num_pizzas * avg_pizzas,
                "price": avg_ingredient_price // 2,
            },
        },
        "pizzas": {},
    }

    for i in range(num_ingredients - 2):
        model["inventory"][f"ingredient{i}"] = {
            "stock": vary(avg_ingredient_stock),
            "price": vary(avg_ingredient_price),
        }

    for p in range(num_pizzas):
        model["pizzas"][f"pizza{p}"] = {
            "ingredients": {"dough": 1, "tomato": 1},
            "minimum_quantity": vary(avg_min_pizzas),
            "price": vary(avg_price),
        }
        for j in range(vary(avg_ingredients - 2)):
            i = r.randrange(0, num_ingredients - 2)
            model["pizzas"][f"pizza{p}"]["ingredients"][f"ingredient{i}"] = r.randrange(
                1, 3
            )

    return model
