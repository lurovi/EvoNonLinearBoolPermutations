import statistics

import numpy as np
import random
from prettytable import PrettyTable
from collections.abc import Callable
from typing import Any
from selection import tournament
from walsh_transform import WalshTransform

from generator import Individual
from cellular.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from cellular.support import compute_all_possible_neighborhoods, create_neighbors_topology_factory, global_moran_I, compute_euclidean_diversity_all_distinct_distances, one_matrix_zero_diagonal, simple_selection_process, weights_matrix_for_morans_I


def check_all_truth_tables_are_balanced(truth_tables: list[np.ndarray]) -> bool:
    """
    Check if all truth tables in the list are balanced.
    """
    for tt in truth_tables:
        if np.sum(tt) != len(tt) // 2:
            return False
    return True


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
        walsh: WalshTransform,
        pop_size: int,
        n_iter: int,
        initialize: Callable[[int], list[Any]],
        evaluate: Callable[[Any], float],
        mate: Callable[[Any, Any], Any],
        mutate: Callable[[Any], Any],
        equal_individuals: Callable[[Any, Any], bool],
        rng: np.random.Generator,
        rand: random.Random,
        cx_rate: float = 0.6,
        mut_rate: float = 0.2,
        verbose: bool = False,
        plateau_iter: int = 1000000,
        mutually_exclusive: bool = False,
        affinity_function: Callable[[Individual, Individual], float] = None,
        matchmaker_pool_rate: float = 0.8,
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
    spectra = [walsh.apply(ind) for ind in temp_pop]
    population = [Individual(ind, evaluate(spectrum[0]), spectrum[0], spectrum[1]) for ind, spectrum in zip(temp_pop, spectra)]

    best_idx = int(np.argmax([ind.fitness for ind in population]))
    best, best_score = population[best_idx], population[best_idx].fitness

    count_plateau = 0

    history = dict()

    history["best_fitness"] = [best_score]
    history["pop_med_fitness"] = [statistics.median([ind.fitness for ind in population])]
    history["pop_q1_fitness"] = [np.percentile([ind.fitness for ind in population], 25)]
    history["pop_q3_fitness"] = [np.percentile([ind.fitness for ind in population], 75)]
    history["pop_mean_fitness"] = [np.mean([ind.fitness for ind in population])]
    history["pop_std_fitness"] = [np.std([ind.fitness for ind in population])]
    history['pop_min_fitness'] = [np.min([ind.fitness for ind in population])]
    history['pop_max_fitness'] = [np.max([ind.fitness for ind in population])]

    history["best_spectral_radius"] = [best.spectral_radius]
    history["best_resiliency"] = [walsh.resiliency(best.spectrum)]
    history["best_correlation_immunity"] = [walsh.correlation_immunity(best.spectrum)]
    history["best_algebraic_degree"] = [walsh.domain().degree(best.genome)[1]]
    history["best_max_autocorrelation_coefficient"] = [walsh.invert(best.spectrum)[1]]

    #history['vanilla_global_moran_I'] = [global_moran_I([ind.spectrum for ind in population], w=one_matrix_zero_diagonal(pop_size))]
    history['real_global_moran_I'] = [global_moran_I([ind.spectrum for ind in population], w=weights_matrix_moran)]
    #history['diversity_median'] = [compute_euclidean_diversity_all_distinct_distances([ind.spectrum for ind in population], measure='median')]

    if isinstance(population[0].genome, np.ndarray) and not check_all_truth_tables_are_balanced([ind.genome for ind in population]):
        raise ValueError(f"Not all truth tables are balanced. Gen {"Initialization"}.")

    for curr_iter in range(n_iter - 1):
        indexed_population = [(i, population[i]) for i in range(pop_size)]
        neighbors_topology = neighbors_topology_factory.create(indexed_population, clone=False)
        current_coordinate_index = 0

        new_population = []
        affinity_values_cache = dict()

        if count_plateau >= plateau_iter:
            temp_pop = initialize(pop_size - 1)
            spectra = [walsh.apply(ind) for ind in temp_pop]
            new_population = [Individual(ind, evaluate(spectrum[0]), spectrum[0], spectrum[1]) for ind, spectrum in zip(temp_pop, spectra)]
            new_population.append(best)
            rand.shuffle(new_population)
            count_plateau = 0
        else:
            while len(new_population) < pop_size:
                # --- Selection ---
                current_coordinate = all_possible_coordinates[current_coordinate_index]
                p1, p2 = simple_selection_process(is_cellular_selection=is_cellular_selection, competitor_rate=cmp_rate, neighbors_topology=neighbors_topology, all_neighborhoods_indices=all_neighborhoods_indices, coordinate=current_coordinate, rand=rand)
                
                # Affinity-based selection of the second parent
                if affinity_function is not None:
                    matchmaker_pool_preliminary = [ind for ind in population if not equal_individuals(ind.genome, p1.genome)]
                    matchmaker_pool = [ind for ind in matchmaker_pool_preliminary if rand.random() < matchmaker_pool_rate]
                    if len(matchmaker_pool) == 0:
                        matchmaker_pool = matchmaker_pool_preliminary
                    if len(matchmaker_pool) != 0:
                        p2 = matchmaker_pool[0]
                        best_affinity = -np.inf
                        for ind in matchmaker_pool:
                            affinity_cache_keys_pair = [str(p1.genome), str(ind.genome)]
                            affinity_cache_keys_pair.sort()
                            affinity_cache_key = (affinity_cache_keys_pair[0], affinity_cache_keys_pair[1])
                            if affinity_cache_key not in affinity_values_cache:
                                affinity_values_cache[affinity_cache_key] = affinity_function(p1, ind)
                            if affinity_values_cache[affinity_cache_key] > best_affinity:
                                p2 = ind
                                best_affinity = affinity_values_cache[affinity_cache_key]
                
                is_changed = False
                p1_genome = p1.genome
                p1_fitness = p1.fitness
                p1_spectrum = p1.spectrum
                p1_spectral_radius = p1.spectral_radius

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

                if is_changed:
                    spectrum = walsh.apply(child)
                    new_population.append(Individual(child, evaluate(spectrum[0]), spectrum[0], spectrum[1]))
                else:
                    new_population.append(Individual(p1_genome, p1_fitness, p1_spectrum, p1_spectral_radius))

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

        history["best_fitness"].append(best_score)
        history["pop_med_fitness"].append(statistics.median([ind.fitness for ind in population]))
        history["pop_q1_fitness"].append(np.percentile([ind.fitness for ind in population], 25))
        history["pop_q3_fitness"].append(np.percentile([ind.fitness for ind in population], 75))
        history["pop_mean_fitness"].append(np.mean([ind.fitness for ind in population]))
        history["pop_std_fitness"].append(np.std([ind.fitness for ind in population]))
        history['pop_min_fitness'].append(np.min([ind.fitness for ind in population]))
        history['pop_max_fitness'].append(np.max([ind.fitness for ind in population]))

        history["best_spectral_radius"].append(best.spectral_radius)
        history["best_resiliency"].append(walsh.resiliency(best.spectrum))
        history["best_correlation_immunity"].append(walsh.correlation_immunity(best.spectrum))
        history["best_algebraic_degree"].append(walsh.domain().degree(best.genome)[1])
        history["best_max_autocorrelation_coefficient"].append(walsh.invert(best.spectrum)[1])

        #history['vanilla_global_moran_I'].append(global_moran_I([ind.spectrum for ind in population], w=one_matrix_zero_diagonal(pop_size)))
        history['real_global_moran_I'].append(global_moran_I([ind.spectrum for ind in population], w=weights_matrix_moran))
        #history['diversity_median'].append(compute_euclidean_diversity_all_distinct_distances([ind.spectrum for ind in population], measure='median'))

        if isinstance(population[0].genome, np.ndarray) and not check_all_truth_tables_are_balanced([ind.genome for ind in population]):
            raise ValueError(f"Not all truth tables are balanced. Gen {curr_iter}.")

    return best, best_score, history
