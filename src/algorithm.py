import statistics

import numpy as np
import random
from prettytable import PrettyTable
from collections.abc import Callable
from typing import Any
from selection import tournament
from walsh_transform import WalshTransform

from generator import Individual, IndividualProgram, clone_program, execute_program, generate_alternate_balanced_binary_vector_one_zero
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


def random_search_truth_tables(
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

    for curr_iter in range(1, n_iter):
        candidate = generate()
        score = evaluate(candidate)
        if score > best_score:
            best_solution, best_score = candidate, score

        if verbose:
            table = PrettyTable(["Iteration", "Current Non-Linearity", "Best Non-Linearity"])
            table.add_row([str(curr_iter), str(score), str(best_score)])
            print(table)

    return best_solution, best_score


def random_search_programs(
        walsh: WalshTransform,
        n_iter: int,
        pipeline_iter_step: int,
        init_bin_size: int,
        generate: Callable[[], Any],
        evaluate: Callable[[Any], float],
        rng: np.random.Generator,
        verbose: bool = False
) -> tuple[Any, float]:
    """
    Random search metaheuristic.
    """
    # Initialize base truth table
    if init_bin_size < 0:
        raise ValueError("init_bin_size must be non-negative.")
    if init_bin_size == 0:
        base_truth_table = generate_alternate_balanced_binary_vector_one_zero(walsh.domain().space_cardinality())
    else:
        # Check if init_bin_size is a power of two
        if (init_bin_size & (init_bin_size - 1)) != 0:
            raise ValueError("init_bin_size must be a power of two.")
        if walsh.domain().space_cardinality() % init_bin_size != 0:
            raise ValueError("init_bin_size must be a divisor of walsh.domain().space_cardinality().")
        # Generate a balanced truth table of size init_bin_size and shuffle it
        temp_base_truth_table = generate_alternate_balanced_binary_vector_one_zero(init_bin_size)
        rng.shuffle(temp_base_truth_table)
        # Replicate the temp_base_truth_table to reach the desired size, which is equal to walsh.domain().space_cardinality()
        base_truth_table = np.tile(temp_base_truth_table, walsh.domain().space_cardinality() // len(temp_base_truth_table))
    
    best_solution = generate()
    genome = execute_program(best_solution, base_truth_table)
    spectrum, _ = walsh.apply(genome)
    best_score = evaluate(spectrum)
    the_global_must_be_updated = True
    global_program = []

    for curr_iter in range(1, n_iter):
        candidate = generate()
        genome = execute_program(candidate, base_truth_table)
        spectrum, _ = walsh.apply(genome)
        score = evaluate(spectrum)
        if score > best_score:
            best_solution, best_score = candidate, score
            the_global_must_be_updated = True

        if verbose:
            table = PrettyTable(["Iteration", "Current Non-Linearity", "Best Non-Linearity"])
            table.add_row([str(curr_iter), str(score), str(best_score)])
            print(table)
            
        if the_global_must_be_updated and (curr_iter + 1) % pipeline_iter_step == 0:
            global_program = clone_program(global_program + best_solution)
            base_truth_table = execute_program(best_solution, base_truth_table).copy()
            the_global_must_be_updated = False

    return best_solution, best_score


def simulated_annealing_truth_tables(
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

    for curr_iter in range(1, n_iter):
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


def simulated_annealing_programs(
        walsh: WalshTransform,
        n_iter: int,
        pipeline_iter_step: int,
        init_bin_size: int,
        generate: Callable[[], Any],
        evaluate: Callable[[Any], float],
        mutate: Callable[[Any], Any],
        rand: random.Random,
        rng: np.random.Generator,
        T0: float = 1.0,
        alpha: float = 0.99,
        verbose: bool = False
) -> tuple[Any, float]:
    """
    Simulated Annealing.
    """
    # Initialize base truth table
    if init_bin_size < 0:
        raise ValueError("init_bin_size must be non-negative.")
    if init_bin_size == 0:
        base_truth_table = generate_alternate_balanced_binary_vector_one_zero(walsh.domain().space_cardinality())
    else:
        # Check if init_bin_size is a power of two
        if (init_bin_size & (init_bin_size - 1)) != 0:
            raise ValueError("init_bin_size must be a power of two.")
        if walsh.domain().space_cardinality() % init_bin_size != 0:
            raise ValueError("init_bin_size must be a divisor of walsh.domain().space_cardinality().")
        # Generate a balanced truth table of size init_bin_size and shuffle it
        temp_base_truth_table = generate_alternate_balanced_binary_vector_one_zero(init_bin_size)
        rng.shuffle(temp_base_truth_table)
        # Replicate the temp_base_truth_table to reach the desired size, which is equal to walsh.domain().space_cardinality()
        base_truth_table = np.tile(temp_base_truth_table, walsh.domain().space_cardinality() // len(temp_base_truth_table))

    current = generate()
    genome = execute_program(current, base_truth_table)
    spectrum, _ = walsh.apply(genome)
    current_score = evaluate(spectrum)
    the_global_must_be_updated = True
    global_program = []
    best, best_score = current, current_score
    T = T0

    for curr_iter in range(1, n_iter):
        candidate = mutate(current)
        genome = execute_program(candidate, base_truth_table)
        spectrum, _ = walsh.apply(genome)
        score = evaluate(spectrum)
        delta = score - current_score

        if delta > 0.0 or rand.random() < np.exp(delta / T):
            current, current_score = candidate, score
            if score > best_score:
                best, best_score = candidate, score
                the_global_must_be_updated = True
        T *= alpha

        if verbose:
            table = PrettyTable(["Iteration", "Current Non-Linearity", "Best Non-Linearity"])
            table.add_row([str(curr_iter), str(score), str(best_score)])
            print(table)

        if the_global_must_be_updated and (curr_iter + 1) % pipeline_iter_step == 0:
            global_program = clone_program(global_program + best)
            base_truth_table = execute_program(best, base_truth_table).copy()
            the_global_must_be_updated = False

    return best, best_score


def evolutionary_algorithm_truth_tables(
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
        cx_rate: float,
        mut_rate: float,
        save_fitness_list_for_each_gen: bool,
        verbose: bool,
        plateau_iter: int,
        mutually_exclusive: bool,
        duplicates_elimination_retry: int,
        affinity_function: Callable[[Individual, Individual], float] | None,
        matchmaker_pool_rate: float,
        # Cellular GA parameters
        pressure: int,
        torus_dim: int,
        radius: int,
        pop_shape: tuple[int, ...],
        cmp_rate: float,
) -> tuple[Any, float, dict[str, Any]]:
    """
    Evolutionary algorithm with elitism, single-child crossover, and mutation.
    """
    
    is_cellular_selection = torus_dim != 0
    neighbors_topology_factory = create_neighbors_topology_factory(pop_size=pop_size, pop_shape=pop_shape, torus_dim=torus_dim, radius=radius, pressure=pressure, rand=rand)

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

    if save_fitness_list_for_each_gen:
        history['pop_fitness_list'] = [" ".join([str(ind.fitness) for ind in population])]

    if isinstance(population[0].genome, np.ndarray) and not check_all_truth_tables_are_balanced([ind.genome for ind in population]):
        raise ValueError(f"Not all truth tables are balanced. Gen {'Initialization'}.")

    for curr_iter in range(1, n_iter):
        indexed_population = [(i, population[i]) for i in range(pop_size)]
        neighbors_topology = neighbors_topology_factory.create(indexed_population, clone=False)
        current_coordinate_index = 0

        new_population = []
        affinity_values_cache = dict()
        already_seen = set()

        if count_plateau >= plateau_iter:
            temp_pop = initialize(pop_size - 1)
            spectra = [walsh.apply(ind) for ind in temp_pop]
            new_population = [Individual(ind, evaluate(spectrum[0]), spectrum[0], spectrum[1]) for ind, spectrum in zip(temp_pop, spectra)]
            new_population.append(best)
            rand.shuffle(new_population)
            count_plateau = 0
        else:
            while len(new_population) < pop_size:
                current_coordinate = all_possible_coordinates[current_coordinate_index]
                
                # Retry mechanism for duplicates elimination
                retry_count = 0
                while True:
                    # --- Selection ---
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

                    child_as_str = str(child)
                    if duplicates_elimination_retry <= 0 or (duplicates_elimination_retry > 0 and child_as_str not in already_seen):
                        already_seen.add(child_as_str)
                        break
                    else:
                        retry_count += 1
                        if retry_count >= duplicates_elimination_retry:
                            # Accept duplicate after max retries
                            break

                if is_changed:
                    spectrum = walsh.apply(child)
                    new_individual = Individual(child, evaluate(spectrum[0]), spectrum[0], spectrum[1])
                else:
                    new_individual = Individual(p1_genome, p1_fitness, p1_spectrum, p1_spectral_radius)
                
                new_population.append(new_individual)
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

        if save_fitness_list_for_each_gen:
            history['pop_fitness_list'].append(" ".join([str(ind.fitness) for ind in population]))

        if isinstance(population[0].genome, np.ndarray) and not check_all_truth_tables_are_balanced([ind.genome for ind in population]):
            raise ValueError(f"Not all truth tables are balanced. Gen {curr_iter}.")

    return best, best_score, history


def evolutionary_algorithm_programs(
        walsh: WalshTransform,
        init_bin_size: int,
        pop_size: int,
        n_iter: int,
        pipeline_iter_step: int,
        initialize: Callable[[int], list[Any]],
        evaluate: Callable[[Any], float],
        mate: Callable[[Any, Any], Any],
        mutate: Callable[[Any], Any],
        equal_individuals: Callable[[Any, Any], bool],
        rng: np.random.Generator,
        rand: random.Random,
        cx_rate: float,
        mut_rate: float,
        save_fitness_list_for_each_gen: bool,
        verbose: bool,
        plateau_iter: int,
        mutually_exclusive: bool,
        duplicates_elimination_retry: int,
        affinity_function: Callable[[IndividualProgram, IndividualProgram], float] | None,
        matchmaker_pool_rate: float,
        # Cellular GA parameters
        pressure: int,
        torus_dim: int,
        radius: int,
        pop_shape: tuple[int, ...],
        cmp_rate: float,
) -> tuple[Any, np.ndarray, float, dict[str, Any]]:
    """
    Evolutionary algorithm with elitism, single-child crossover, and mutation.
    """

    is_cellular_selection = torus_dim != 0
    neighbors_topology_factory = create_neighbors_topology_factory(pop_size=pop_size, pop_shape=pop_shape, torus_dim=torus_dim, radius=radius, pressure=pressure, rand=rand)

    all_possible_coordinates, all_neighborhoods_indices = compute_all_possible_neighborhoods(pop_size=pop_size, pop_shape=pop_shape, is_cellular_selection=is_cellular_selection, neighbors_topology_factory=neighbors_topology_factory)
    weights_matrix_moran = weights_matrix_for_morans_I(pop_size=pop_size, is_cellular_selection=is_cellular_selection, all_possible_coordinates=all_possible_coordinates, all_neighborhoods_indices=all_neighborhoods_indices)

    temp_base_truth_table_as_str = "1010" # Default value in case init_bin_size == 0

    # Initialize base truth table
    if init_bin_size < 0:
        raise ValueError("init_bin_size must be non-negative.")
    if init_bin_size == 0:
        base_truth_table = generate_alternate_balanced_binary_vector_one_zero(walsh.domain().space_cardinality())
    else:
        # Check if init_bin_size is a power of two
        if (init_bin_size & (init_bin_size - 1)) != 0:
            raise ValueError("init_bin_size must be a power of two.")
        if walsh.domain().space_cardinality() % init_bin_size != 0:
            raise ValueError("init_bin_size must be a divisor of walsh.domain().space_cardinality().")
        # Generate a balanced truth table of size init_bin_size and shuffle it
        temp_base_truth_table = generate_alternate_balanced_binary_vector_one_zero(init_bin_size)
        rng.shuffle(temp_base_truth_table)
        # Replicate the temp_base_truth_table to reach the desired size, which is equal to walsh.domain().space_cardinality()
        base_truth_table = np.tile(temp_base_truth_table, walsh.domain().space_cardinality() // len(temp_base_truth_table))
        temp_base_truth_table_as_str = "".join([str(aaa) for aaa in temp_base_truth_table.tolist()])

    original_truth_table = base_truth_table.copy()
    global_program = []
    the_global_must_be_updated = True

    # Initialize population
    temp_pop = initialize(pop_size)
    truth_tables = [execute_program(prog, base_truth_table) for prog in temp_pop]
    spectra = [walsh.apply(tt) for tt in truth_tables]
    population = [IndividualProgram(ind, base_truth_table, genome, evaluate(spectrum[0]), spectrum[0], spectrum[1]) for ind, genome, spectrum in zip(temp_pop, truth_tables, spectra)]

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

    history['global_program_length'] = [len(global_program)]
    history["best_length"] = [len(best.program)]
    history["pop_med_length"] = [statistics.median([len(ind.program) for ind in population])]
    history["pop_q1_length"] = [np.percentile([len(ind.program) for ind in population], 25)]
    history["pop_q3_length"] = [np.percentile([len(ind.program) for ind in population], 75)]
    history["pop_mean_length"] = [np.mean([len(ind.program) for ind in population])]
    history["pop_std_length"] = [np.std([len(ind.program) for ind in population])]
    history['pop_min_length'] = [np.min([len(ind.program) for ind in population])]
    history['pop_max_length'] = [np.max([len(ind.program) for ind in population])]

    history["best_spectral_radius"] = [best.spectral_radius]
    history["best_resiliency"] = [walsh.resiliency(best.spectrum)]
    history["best_correlation_immunity"] = [walsh.correlation_immunity(best.spectrum)]
    history["best_algebraic_degree"] = [walsh.domain().degree(best.genome)[1]]
    history["best_max_autocorrelation_coefficient"] = [walsh.invert(best.spectrum)[1]]

    #history['vanilla_global_moran_I'] = [global_moran_I([ind.spectrum for ind in population], w=one_matrix_zero_diagonal(pop_size))]
    history['real_global_moran_I'] = [global_moran_I([ind.spectrum for ind in population], w=weights_matrix_moran)]
    #history['diversity_median'] = [compute_euclidean_diversity_all_distinct_distances([ind.spectrum for ind in population], measure='median')]

    history["initial_truth_table_block_str"] = [temp_base_truth_table_as_str]

    if save_fitness_list_for_each_gen:
        history['pop_fitness_list'] = [" ".join([str(ind.fitness) for ind in population])]

    if isinstance(population[0].genome, np.ndarray) and not check_all_truth_tables_are_balanced([ind.genome for ind in population]):
        raise ValueError(f"Not all truth tables are balanced. Gen {'Initialization'}.")

    for curr_iter in range(1, n_iter):
        indexed_population = [(i, population[i]) for i in range(pop_size)]
        neighbors_topology = neighbors_topology_factory.create(indexed_population, clone=False)
        current_coordinate_index = 0

        new_population = []
        affinity_values_cache = dict()

        if count_plateau >= plateau_iter:
            temp_pop = initialize(pop_size - 1)
            truth_tables = [execute_program(prog, base_truth_table) for prog in temp_pop]
            spectra = [walsh.apply(tt) for tt in truth_tables]
            new_population = [IndividualProgram(ind, base_truth_table, genome, evaluate(spectrum[0]), spectrum[0], spectrum[1]) for ind, genome, spectrum in zip(temp_pop, truth_tables, spectra)]
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
                    matchmaker_pool_preliminary = [ind for ind in population if not equal_individuals(ind.program, p1.program)]
                    matchmaker_pool = [ind for ind in matchmaker_pool_preliminary if rand.random() < matchmaker_pool_rate]
                    if len(matchmaker_pool) == 0:
                        matchmaker_pool = matchmaker_pool_preliminary
                    if len(matchmaker_pool) != 0:
                        p2 = matchmaker_pool[0]
                        best_affinity = -np.inf
                        for ind in matchmaker_pool:
                            affinity_cache_keys_pair = [str(p1.program), str(ind.program)]
                            affinity_cache_keys_pair.sort()
                            affinity_cache_key = (affinity_cache_keys_pair[0], affinity_cache_keys_pair[1])
                            if affinity_cache_key not in affinity_values_cache:
                                affinity_values_cache[affinity_cache_key] = affinity_function(p1, ind)
                            if affinity_values_cache[affinity_cache_key] > best_affinity:
                                p2 = ind
                                best_affinity = affinity_values_cache[affinity_cache_key]
                
                is_changed = False
                p1_program = p1.program
                p1_base_genome = p1.base_genome
                p1_genome = p1.genome
                p1_fitness = p1.fitness
                p1_spectrum = p1.spectrum
                p1_spectral_radius = p1.spectral_radius

                # --- Variation ---
                if not mutually_exclusive:
                    # crossover
                    if rand.random() < cx_rate:
                        child = mate(p1.program, p2.program)
                        is_changed = True
                    else:
                        child = p1.program

                    # mutation
                    if rand.random() < mut_rate:
                        child = mutate(child)
                        is_changed = True
                else:
                    is_changed = True
                    # crossover
                    if rand.random() < cx_rate / (cx_rate + mut_rate):
                        child = mate(p1.program, p2.program)
                    # mutation
                    else:
                        child = mutate(p1.program)

                if is_changed:
                    truth_table = execute_program(child, base_truth_table)
                    spectrum = walsh.apply(truth_table)
                    new_population.append(IndividualProgram(child, base_truth_table, truth_table, evaluate(spectrum[0]), spectrum[0], spectrum[1]))
                else:
                    new_population.append(IndividualProgram(p1_program, p1_base_genome, p1_genome, p1_fitness, p1_spectrum, p1_spectral_radius))

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
            the_global_must_be_updated = True
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

        history['global_program_length'].append(len(global_program))
        history["best_length"].append(len(best.program))
        history["pop_med_length"].append(statistics.median([len(ind.program) for ind in population]))
        history["pop_q1_length"].append(np.percentile([len(ind.program) for ind in population], 25))
        history["pop_q3_length"].append(np.percentile([len(ind.program) for ind in population], 75))
        history["pop_mean_length"].append(np.mean([len(ind.program) for ind in population]))
        history["pop_std_length"].append(np.std([len(ind.program) for ind in population]))
        history['pop_min_length'].append(np.min([len(ind.program) for ind in population]))
        history['pop_max_length'].append(np.max([len(ind.program) for ind in population]))

        history["best_spectral_radius"].append(best.spectral_radius)
        history["best_resiliency"].append(walsh.resiliency(best.spectrum))
        history["best_correlation_immunity"].append(walsh.correlation_immunity(best.spectrum))
        history["best_algebraic_degree"].append(walsh.domain().degree(best.genome)[1])
        history["best_max_autocorrelation_coefficient"].append(walsh.invert(best.spectrum)[1])

        #history['vanilla_global_moran_I'].append(global_moran_I([ind.spectrum for ind in population], w=one_matrix_zero_diagonal(pop_size)))
        history['real_global_moran_I'].append(global_moran_I([ind.spectrum for ind in population], w=weights_matrix_moran))
        #history['diversity_median'].append(compute_euclidean_diversity_all_distinct_distances([ind.spectrum for ind in population], measure='median'))

        history["initial_truth_table_block_str"].append(temp_base_truth_table_as_str)

        if save_fitness_list_for_each_gen:
            history['pop_fitness_list'].append(" ".join([str(ind.fitness) for ind in population]))

        if isinstance(population[0].genome, np.ndarray) and not check_all_truth_tables_are_balanced([ind.genome for ind in population]):
            raise ValueError(f"Not all truth tables are balanced. Gen {curr_iter}.")
        
        if the_global_must_be_updated and (curr_iter + 1) % pipeline_iter_step == 0:
            global_program = clone_program(global_program + best.program)
            base_truth_table = execute_program(best.program, base_truth_table).copy()
            the_global_must_be_updated = False

    # Check that the truth table of the best individual is the same as the one obtained by executing the global program on the original truth table
    checked_truth_table = execute_program(global_program, original_truth_table)
    if not np.array_equal(best.genome, checked_truth_table):
        raise ValueError("The truth table of the best individual is not the same as the one obtained by executing the global program on the original truth table.")

    return global_program, original_truth_table, best_score, history
