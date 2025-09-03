import random


def tournament[T](population: list[T], scores: list[float], tournament_size: int, rand: random.Random) -> T:
    paired = list(zip(population, scores))
    tour = rand.choices(paired, k=tournament_size)
    best = max(tour, key=lambda x: x[1])
    return best[0]
