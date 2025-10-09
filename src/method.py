from generator import *
from selection import *
from crossover import *
from mutation import *
from distance import *
from full_binary_domain import FullBinaryDomain
from algorithm import random_search_programs, random_search_truth_tables, simulated_annealing_truth_tables, simulated_annealing_programs, evolutionary_algorithm_truth_tables, evolutionary_algorithm_programs
from cellular.support import compute_all_possible_neighborhoods, create_neighbors_topology_factory, weights_matrix_for_morans_I
from walsh_transform import WalshTransform


# ===========================================================================
# GA Truth Table Permutations Methods
# ===========================================================================

def truth_tables_rs(
        n_bits: int,
        n_iter: int,
        seed: int,
        verbose: bool = False
):
    rng = np.random.default_rng(seed)

    domain = FullBinaryDomain(n_bits)
    walsh = WalshTransform(n_bits)

    generate = lambda: generate_random_balanced_binary_vector(domain.space_cardinality(), rng)
    evaluate = lambda x: walsh.granular_non_linearity(walsh.apply(x)[0])

    best_solution, best_score = random_search_truth_tables(
        n_iter=n_iter,
        generate=generate,
        evaluate=evaluate,
        verbose=verbose,
    )

    return best_solution, best_score


def truth_tables_sa(
        n_bits: int,
        n_iter: int,
        seed: int,
        verbose: bool = False,
):
    rng = np.random.default_rng(seed)
    rand = random.Random(seed)

    domain = FullBinaryDomain(n_bits)
    walsh = WalshTransform(n_bits)

    generate = lambda: generate_random_balanced_binary_vector(domain.space_cardinality(), rng)
    evaluate = lambda x: walsh.granular_non_linearity(walsh.apply(x)[0])
    mutate = lambda x: consecutive_swap_mutation(x, 1, 20, rng)

    best_solution, best_score = simulated_annealing_truth_tables(
        n_iter=n_iter,
        generate=generate,
        evaluate=evaluate,
        mutate=mutate,
        rand=rand,
        verbose=verbose,
    )

    return best_solution, best_score


def truth_tables_ea(
        n_bits: int,
        pop_size: int,
        n_iter: int,
        seed: int,
        pressure: int,
        torus_dim: int,
        radius: int,
        pop_shape: tuple[int, ...],
        cmp_rate: float,
        matchmaker_pool_rate: float,
        affinity_type: str,
        verbose: bool = False
):
    rng = np.random.default_rng(seed)
    rand = random.Random(seed)

    domain = FullBinaryDomain(n_bits)
    walsh = WalshTransform(n_bits)

    if affinity_type == 'random':
        affinity_function = None
    elif affinity_type == 'hamming_similarity':
        affinity_function = hamming_similarity
    elif affinity_type == 'jaccard_similarity':
        affinity_function = jaccard_similarity
    elif affinity_type == 'jaccard_distance':
        affinity_function = jaccard_distance
    elif affinity_type == 'hamming_distance':
        affinity_function = hamming_distance
    elif affinity_type == 'nearest_fitness':
        affinity_function = inv_absolute_difference_between_numbers
    elif affinity_type == 'furthest_fitness':
        affinity_function = absolute_difference_between_numbers
    elif affinity_type == 'spectrum_distance':
        affinity_function = lambda x, y: walsh_spectral_distance(x, y, walsh)
    elif affinity_type == 'spectrum_similarity':
        affinity_function = lambda x, y: walsh_spectral_similarity(x, y, walsh)
    elif affinity_type == 'abs_spectrum_distance':
        affinity_function = lambda x, y: abs_walsh_spectral_distance(x, y, walsh)
    elif affinity_type == 'abs_spectrum_similarity':
        affinity_function = lambda x, y: abs_walsh_spectral_similarity(x, y, walsh)
    else:
        raise ValueError(f"Unknown affinity type: {affinity_type}")
    
    initialize = lambda x: uniform_initial_population(domain.space_cardinality(), x, rng, rand)
    #initialize = lambda x: greedy_maxmin_initial_population(domain.space_cardinality(), x, rng, rand)
    evaluate = lambda x: walsh.granular_non_linearity(x)
    
    mate = lambda x, y: position_based_crossover(x, y, rng)
    #mate = lambda x, y: uniform_crossover_with_repair(x, y, rng)
    #mate = lambda x, y: order_crossover(x, y, rng)
    #mate = lambda x, y: cycle_crossover_binary(x, y, rng)
    #mate = lambda x, y: pmx_binary(x, y, rng)

    mutate = lambda x: consecutive_swap_mutation(x, 1, 5, rng)
    #mutate = lambda x: local_search_swaps(x, evaluate, rand, eval_budget=10)
    # mutate = lambda x: spectrum_aware_swap(
    #     ind=x,
    #     walsh=walsh,
    #     evaluate=evaluate,
    #     rand=rand,
    #     eval_budget=10,
    #     top_frac=0.1,
    # )
    # mutate = lambda x: programs_random_search_enhancing_balanced_truth_table(
    #     domain=domain,
    #     walsh=walsh,
    #     ind=x,
    #     eval_budget=10,
    #     min_length=2,
    #     max_length=10,
    #     sampling_probabilities=[1.0] + [0.0] * 6,
    #     rng=rng,
    #     rand=rand,
    # )
    # mutate = lambda x: programs_simulated_annealing_enhancing_balanced_truth_table(
    #     domain=domain,
    #     walsh=walsh,
    #     ind=x,
    #     eval_budget=10,
    #     min_length=2,
    #     max_length=10,
    #     sampling_probabilities=[1.0] + [0.0] * 6,
    #     rng=rng,
    #     rand=rand,
    # )
    # mutate = lambda x: programs_evolutionary_algorithm_enhancing_balanced_truth_table(
    #     domain=domain,
    #     walsh=walsh,
    #     ind=x,
    #     pop_size=2,
    #     n_iter=5,
    #     min_length=2,
    #     max_length=10,
    #     sampling_probabilities=[1.0] + [0.0] * 6,
    #     rng=rng,
    #     rand=rand,
    # )

    best_solution, best_score, history = evolutionary_algorithm_truth_tables(
        walsh=walsh,
        pop_size=pop_size,
        n_iter=n_iter,
        initialize=initialize,
        evaluate=evaluate,
        mate=mate,
        mutate=mutate,
        rng=rng,
        rand=rand,
        verbose=verbose,
        cx_rate=0.5,
        mut_rate=0.5,
        mutually_exclusive=True,
        plateau_iter=50 * 10_000,
        # Affinity parameters
        affinity_function=affinity_function,
        matchmaker_pool_rate=matchmaker_pool_rate,
        # Cellular GA parameters
        pressure=pressure,
        torus_dim=torus_dim,
        radius=radius,
        pop_shape=pop_shape,
        cmp_rate=cmp_rate,
        equal_individuals=lambda a, b: np.array_equal(a, b)
    )

    return best_solution, best_score, history


# ===========================================================================
# Linear GP Truth Table Program Permutations Methods
# ===========================================================================

def programs_rs(
        n_bits: int,
        n_iter: int,
        pipeline_iter_step: int,
        init_bin_size: int,
        seed: int,
        min_length: int,
        max_length: int,
        sampling_probabilities: list[float],
        verbose: bool = False
):
    if sum(sampling_probabilities) <= 0:
        raise ValueError("At least one sampling probability must be greater than zero.")
    if abs(sum(sampling_probabilities) - 1.0) > 1e-8:
        raise ValueError("Sampling probabilities must sum to 1.0.")    

    rng = np.random.default_rng(seed)
    rand = random.Random(seed)

    domain = FullBinaryDomain(n_bits)
    walsh = WalshTransform(n_bits)

    generate = lambda: random_program(domain.space_cardinality(), sampling_probabilities, min_length, max_length, rng, rand)
    evaluate = lambda x: walsh.granular_non_linearity(x)

    best_solution, best_score = random_search_programs(
        walsh=walsh,
        init_bin_size=init_bin_size,
        pipeline_iter_step=pipeline_iter_step,
        n_iter=n_iter,
        generate=generate,
        evaluate=evaluate,
        rng=rng,
        verbose=verbose,
    )

    return best_solution, best_score


def programs_sa(
        n_bits: int,
        n_iter: int,
        pipeline_iter_step: int,
        init_bin_size: int,
        seed: int,
        min_length: int,
        max_length: int,
        sampling_probabilities: list[float],
        verbose: bool = False,
):
    if sum(sampling_probabilities) <= 0:
        raise ValueError("At least one sampling probability must be greater than zero.")
    if abs(sum(sampling_probabilities) - 1.0) > 1e-8:
        raise ValueError("Sampling probabilities must sum to 1.0.")
    
    rng = np.random.default_rng(seed)
    rand = random.Random(seed)

    domain = FullBinaryDomain(n_bits)
    walsh = WalshTransform(n_bits)

    generate = lambda: random_program(domain.space_cardinality(), sampling_probabilities, min_length, max_length, rng, rand)
    evaluate = lambda x: walsh.granular_non_linearity(x)
    mutate = lambda x: mutate_program(x, domain.space_cardinality(), sampling_probabilities, min_length, max_length, rand)

    best_solution, best_score = simulated_annealing_programs(
        walsh=walsh,
        n_iter=n_iter,
        pipeline_iter_step=pipeline_iter_step,
        init_bin_size=init_bin_size,
        generate=generate,
        evaluate=evaluate,
        mutate=mutate,
        rand=rand,
        rng=rng,
        verbose=verbose,
    )

    return best_solution, best_score


def programs_ea(
        n_bits: int,
        init_bin_size: int,
        pop_size: int,
        #warm_up: int,
        n_iter: int,
        pipeline_iter_step: int,
        seed: int,
        min_length: int,
        max_length: int,
        sampling_probabilities: list[float],
        pressure: int,
        torus_dim: int,
        radius: int,
        pop_shape: tuple[int, ...],
        cmp_rate: float,
        matchmaker_pool_rate: float,
        affinity_type: str,
        verbose: bool = False,
):
    if sum(sampling_probabilities) <= 0:
        raise ValueError("At least one sampling probability must be greater than zero.")
    if abs(sum(sampling_probabilities) - 1.0) > 1e-8:
        raise ValueError("Sampling probabilities must sum to 1.0.")
    
    rng = np.random.default_rng(seed)
    rand = random.Random(seed)

    domain = FullBinaryDomain(n_bits)
    walsh = WalshTransform(n_bits)

    #base_truth_table = truth_tables_ea(n_bits, pop_size, warm_up, seed, pressure=pressure, torus_dim=torus_dim, radius=radius, pop_shape=pop_shape, cmp_rate=cmp_rate, verbose=False)[0].genome

    if affinity_type == 'random':
        affinity_function = None
    # elif affinity_type == 'hamming_similarity':
    #     affinity_function = hamming_similarity
    # elif affinity_type == 'jaccard_similarity':
    #     affinity_function = jaccard_similarity
    # elif affinity_type == 'jaccard_distance':
    #     affinity_function = jaccard_distance
    # elif affinity_type == 'hamming_distance':
    #     affinity_function = hamming_distance
    # elif affinity_type == 'nearest_fitness':
    #     affinity_function = inv_absolute_difference_between_numbers
    # elif affinity_type == 'furthest_fitness':
    #     affinity_function = absolute_difference_between_numbers
    else:
        raise ValueError(f"Unknown affinity type: {affinity_type}")

    initialize = lambda x: initialize_population_programs(x, domain.space_cardinality(), sampling_probabilities, min_length, max_length, rng, rand)
    evaluate = lambda x: walsh.granular_non_linearity(x)
    mate = lambda x, y: homologous_crossover(x, y, min_length, max_length, rand)
    mutate = lambda x: mutate_program(x, domain.space_cardinality(), sampling_probabilities, min_length, max_length, rand)

    global_program, original_truth_table, best_score, history = evolutionary_algorithm_programs(
        walsh=walsh,
        init_bin_size=init_bin_size,
        pop_size=pop_size,
        n_iter=n_iter,
        pipeline_iter_step=pipeline_iter_step,
        initialize=initialize,
        evaluate=evaluate,
        mate=mate,
        mutate=mutate,
        rng=rng,
        rand=rand,
        verbose=verbose,
        cx_rate=0.5,
        mut_rate=0.5,
        mutually_exclusive=True,
        plateau_iter=50 * 10_000,
        # Affinity parameters
        affinity_function=affinity_function,
        matchmaker_pool_rate=matchmaker_pool_rate,
        # Cellular GA parameters
        pressure=pressure,
        torus_dim=torus_dim,
        radius=radius,
        pop_shape=pop_shape,
        cmp_rate=cmp_rate,
        equal_individuals=lambda a, b: a == b
    )

    return global_program, original_truth_table, best_score, history
