import statistics

import numpy as np
import random
from prettytable import PrettyTable
from collections.abc import Callable
from typing import Any

from cellular.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from cellular.support import compute_all_possible_neighborhoods, create_neighbors_topology_factory, simple_selection_process, weights_matrix_for_morans_I


class Individual:
    def __init__(self, genome: Any, fitness: float):
        self.genome = genome
        self.fitness = fitness


def random_search(
        n_iter: int,
        generate: Callable[[], Any],
        evaluate: Callable[[Any], float],
        verbose: bool = False
) -> tuple[Any, float]:
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
        generate: Callable[[], Any],
        evaluate: Callable[[Any], float],
        mutate: Callable[[Any], Any],
        rand: random.Random,
        T0: float = 1.0,
        alpha: float = 0.99,
        verbose: bool = False
) -> tuple[Any, float]:
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
        initialize: Callable[[int], list[Any]],
        evaluate: Callable[[Any], float],
        mate: Callable[[Any, Any], Any],
        mutate: Callable[[Any], Any],
        rng: np.random.Generator,
        rand: random.Random,
        cx_rate: float = 0.6,
        mut_rate: float = 0.2,
        verbose: bool = False,
        plateau_iter: int = 1000000,
        mutually_exclusive: bool = False,
        # Cellular GA parameters
        pressure: int = 2,
        torus_dim: int = 0,
        radius: int = 0,
        pop_shape: tuple[int, ...] = (),
        cmp_rate: float = 0.0,
) -> tuple[Any, float]:
    """
    Evolutionary algorithm with elitism, single-child crossover, and mutation.
    """
    
    is_cellular_selection = torus_dim != 0
    neighbors_topology_factory = create_neighbors_topology_factory(pop_size=pop_size, pop_shape=pop_shape, torus_dim=torus_dim, radius=radius, pressure=pressure)

    all_possible_coordinates, all_neighborhoods_indices = compute_all_possible_neighborhoods(pop_size=pop_size, pop_shape=pop_shape, is_cellular_selection=is_cellular_selection, neighbors_topology_factory=neighbors_topology_factory)
    weights_matrix_moran = weights_matrix_for_morans_I(pop_size=pop_size, is_cellular_selection=is_cellular_selection, all_possible_coordinates=all_possible_coordinates, all_neighborhoods_indices=all_neighborhoods_indices)
    
    # Initialize population
    temp_pop = initialize(pop_size)
    population = [Individual(ind, evaluate(ind)) for ind in temp_pop]

    best_idx = int(np.argmax([ind.fitness for ind in population]))
    best, best_score = population[best_idx], population[best_idx].fitness

    count_plateau = 0

    for curr_iter in range(n_iter - 1):
        indexed_population = [(i, population[i]) for i in range(pop_size)]
        neighbors_topology = neighbors_topology_factory.create(indexed_population, clone=False)
        current_coordinate_index = 0

        new_population = []

        if count_plateau >= plateau_iter:
            temp_pop = initialize(pop_size - 1)
            new_population = [Individual(ind, evaluate(ind)) for ind in temp_pop]
            new_population.append(best)
            rand.shuffle(new_population)
            count_plateau = 0
        else:
            while len(new_population) < pop_size:
                # --- Selection ---
                current_coordinate = all_possible_coordinates[current_coordinate_index]
                p1, p2 = simple_selection_process(is_cellular_selection=is_cellular_selection, competitor_rate=cmp_rate, neighbors_topology=neighbors_topology, all_neighborhoods_indices=all_neighborhoods_indices, coordinate=current_coordinate)
                is_changed = False
                p1_fitness = p1.fitness
                # --- Variation ---
                if not mutually_exclusive:
                    # crossover
                    if rand.random() < cx_rate:
                        child = mate(p1.genome, p2.genome)
                        is_changed = True
                    else:
                        child = p1.genome

                    # mutation
                    if rand.random() < mut_rate:
                        child = mutate(child)
                        is_changed = True
                else:
                    is_changed = True
                    # crossover
                    if rand.random() < cx_rate / (cx_rate + mut_rate):
                        child = mate(p1.genome, p2.genome)
                    # mutation
                    else:
                        child = mutate(p1.genome)

                new_population.append(Individual(child, evaluate(child) if is_changed else p1_fitness))
                current_coordinate_index += 1

        # --- Elitism ---
        # Keep best-so-far if not already in new population
        worst_idx = int(np.argmin([ind.fitness for ind in new_population]))
        if best_score > new_population[worst_idx].fitness:
            new_population[worst_idx] = best
        # Update best
        gen_best_idx = int(np.argmax([ind.fitness for ind in new_population]))
        gen_best_score = new_population[gen_best_idx].fitness
        if gen_best_score > best_score:
            best, best_score = new_population[gen_best_idx], gen_best_score
            count_plateau = 0
        else:
            count_plateau += 1

        if verbose:
            table = PrettyTable(["Iteration", "Median Non-Linearity", "Current Non-Linearity", "Best Non-Linearity"])
            table.add_row([str(curr_iter), str(statistics.median([ind.fitness for ind in new_population])), str(new_population[gen_best_idx].fitness), str(best_score)])
            print(table)

        # Advance population
        population = new_population

    return best, best_score
