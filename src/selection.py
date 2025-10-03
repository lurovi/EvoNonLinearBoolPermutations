import random
from generator import Individual


def tournament(population: list[Individual], tournament_size: int, rand: random.Random) -> Individual:
    tour = rand.choices(population, k=tournament_size)
    best = max(tour, key=lambda x: x.fitness)
    return best
