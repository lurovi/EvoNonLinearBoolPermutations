import statistics

import numpy as np
import random
from prettytable import PrettyTable
from collections.abc import Callable


def random_search(
        n_iter: int,
        generate: Callable[[], any],
        evaluate: Callable[[any], float],
        verbose: bool = False
) -> tuple[any, float]:
    """
    Random search metaheuristic.
    """
    best_solution = generate()
    best_score = evaluate(best_solution)

    for curr_iter in range(n_iter - 1):
        candidate = generate()
        score = evaluate(candidate)
        if score > best_score:
            best_solution, best_score = candidate, score

        if verbose:
            table = PrettyTable(["Iteration", "Current Non-Linearity", "Best Non-Linearity"])
            table.add_row([str(curr_iter), str(score), str(best_score)])
            print(table)

    return best_solution, best_score


def simulated_annealing(
        n_iter: int,
        generate: Callable[[], any],
        evaluate: Callable[[any], float],
        mutate: Callable[[any], any],
        rand: random.Random,
        T0: float = 1.0,
        alpha: float = 0.99,
        verbose: bool = False
) -> tuple[any, float]:
    """
    Simulated Annealing.
    """
    current = generate()
    current_score = evaluate(current)
    best, best_score = current, current_score
    T = T0

    for curr_iter in range(n_iter - 1):
        candidate = mutate(current)
        score = evaluate(candidate)
        delta = score - current_score

        if delta > 0.0 or rand.random() < np.exp(delta / T):
            current, current_score = candidate, score
            if score > best_score:
                best, best_score = candidate, score
        T *= alpha

        if verbose:
            table = PrettyTable(["Iteration", "Current Non-Linearity", "Best Non-Linearity"])
            table.add_row([str(curr_iter), str(score), str(best_score)])
            print(table)

    return best, best_score


def evolutionary_algorithm(
        pop_size: int,
        n_iter: int,
        generate: Callable[[], any],
        evaluate: Callable[[any], float],
        select: Callable[[list[any], list[float]], any],
        mate: Callable[[any, any], any],
        mutate: Callable[[any], any],
        rng: np.random.Generator,
        rand: random.Random,
        cx_rate: float = 0.6,
        mut_rate: float = 0.2,
        verbose: bool = False,
        plateau_iter: int = 1000000,
        mutually_exclusive: bool = False
) -> tuple[any, float]:
    """
    Evolutionary algorithm with elitism, single-child crossover, and mutation.
    """
    # Initialize population
    population = [generate() for _ in range(pop_size)]
    scores = [evaluate(ind) for ind in population]

    best_idx = int(np.argmax(scores))
    best, best_score = population[best_idx], scores[best_idx]

    count_plateau = 0

    for curr_iter in range(n_iter - 1):
        # --- Selection ---
        if count_plateau >= plateau_iter:
            mating_pool = [generate() for _ in range(pop_size - 1)]
            mating_pool.append(best)
            rand.shuffle(mating_pool)
            count_plateau = 0
        else:
            mating_pool = [select(population, scores) for _ in range(pop_size)]

        is_changed = [False for _ in range(pop_size)]
        new_population = []

        # --- Variation ---
        for j, parent in enumerate(mating_pool, 0):
            # choose random mate
            partner = mating_pool[rng.integers(pop_size)]

            if not mutually_exclusive:
                # crossover
                if rand.random() < cx_rate:
                    is_changed[j] = True
                    child = mate(parent, partner)
                else:
                    child = parent

                # mutation
                if rand.random() < mut_rate:
                    is_changed[j] = True
                    child = mutate(child)
            else:
                is_changed[j] = True
                # crossover
                if rand.random() < cx_rate / (cx_rate + mut_rate):
                    child = mate(parent, partner)
                # mutation
                else:
                    child = mutate(parent)

            new_population.append(child)

        # --- Evaluate new population ---
        scores = [evaluate(ind) if is_changed[i] else scores[i] for i, ind in enumerate(new_population, 0)]

        # --- Elitism ---
        # Keep best-so-far if not already in new population
        worst_idx = int(np.argmin(scores))
        if best_score > scores[worst_idx]:
            new_population[worst_idx] = best
            scores[worst_idx] = best_score

        # Update best
        gen_best_idx = int(np.argmax(scores))
        gen_best_score = scores[gen_best_idx]
        if gen_best_score > best_score:
            best, best_score = new_population[gen_best_idx], gen_best_score
            count_plateau = 0
        else:
            count_plateau += 1

        if verbose:
            table = PrettyTable(["Iteration", "Median Non-Linearity", "Current Non-Linearity", "Best Non-Linearity"])
            table.add_row([str(curr_iter), str(statistics.median(scores)), str(scores[gen_best_idx]), str(best_score)])
            print(table)

        # Advance population
        population = new_population

    return best, best_score
