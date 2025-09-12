from generator import *
from selection import *
from crossover import *
from mutation import *
from full_binary_domain import FullBinaryDomain
from algorithm import random_search, simulated_annealing, evolutionary_algorithm
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

    best_solution, best_score = random_search(
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
    mutate = lambda x: consecutive_swap_mutation(x, 1, domain.space_cardinality() // 4, rng)

    best_solution, best_score = simulated_annealing(
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
        tournament_size: int,
        seed: int,
        verbose: bool = False
):
    rng = np.random.default_rng(seed)
    rand = random.Random(seed)

    domain = FullBinaryDomain(n_bits)
    walsh = WalshTransform(n_bits)

    generate = lambda: generate_random_balanced_binary_vector(domain.space_cardinality(), rng)
    evaluate = lambda x: walsh.granular_non_linearity(walsh.apply(x)[0])
    select = lambda x, y: tournament(x, y, tournament_size, rand)
    mate = lambda x, y: position_based_crossover(x, y, rng)
    mutate = lambda x: consecutive_swap_mutation(x, 1, 5, rng)

    best_solution, best_score = evolutionary_algorithm(
        pop_size=pop_size,
        n_iter=n_iter,
        generate=generate,
        evaluate=evaluate,
        select=select,
        mate=mate,
        mutate=mutate,
        rng=rng,
        rand=rand,
        verbose=verbose,
        cx_rate=0.6,
        mut_rate=0.2,
    )

    return best_solution, best_score


# ===========================================================================
# Linear GP Truth Table Program Permutations Methods
# ===========================================================================

def programs_rs(
        n_bits: int,
        warm_up: int,
        n_iter: int,
        seed: int,
        min_length: int,
        max_length: int,
        verbose: bool = False
):
    rng = np.random.default_rng(seed)
    rand = random.Random(seed)

    domain = FullBinaryDomain(n_bits)
    walsh = WalshTransform(n_bits)

    base_truth_table = generate_alternate_balanced_binary_vector(domain.space_cardinality())
    base_truth_table, _ = truth_tables_rs(n_bits, warm_up, seed, verbose=False)

    generate = lambda: random_program(domain.space_cardinality(), min_length, max_length, rng, rand)
    evaluate = lambda x: walsh.granular_non_linearity(walsh.apply(execute_program(x, base_truth_table))[0])

    best_solution, best_score = random_search(
        n_iter=n_iter - warm_up,
        generate=generate,
        evaluate=evaluate,
        verbose=verbose,
    )

    return best_solution, best_score


def programs_sa(
        n_bits: int,
        warm_up: int,
        n_iter: int,
        seed: int,
        min_length: int,
        max_length: int,
        verbose: bool = False,
):
    rng = np.random.default_rng(seed)
    rand = random.Random(seed)

    domain = FullBinaryDomain(n_bits)
    walsh = WalshTransform(n_bits)

    base_truth_table = generate_alternate_balanced_binary_vector(domain.space_cardinality())
    base_truth_table, _ = truth_tables_sa(n_bits, warm_up, seed, verbose=False)

    generate = lambda: random_program(domain.space_cardinality(), min_length, max_length, rng, rand)
    evaluate = lambda x: walsh.granular_non_linearity(walsh.apply(execute_program(x, base_truth_table))[0])
    mutate = lambda x: mutate_program(x, domain.space_cardinality(), min_length, max_length, rand)

    best_solution, best_score = simulated_annealing(
        n_iter=n_iter - warm_up,
        generate=generate,
        evaluate=evaluate,
        mutate=mutate,
        rand=rand,
        verbose=verbose,
    )

    return best_solution, best_score


def programs_ea(
        n_bits: int,
        pop_size: int,
        warm_up: int,
        n_iter: int,
        tournament_size: int,
        seed: int,
        min_length: int,
        max_length: int,
        verbose: bool = False,
):
    rng = np.random.default_rng(seed)
    rand = random.Random(seed)

    domain = FullBinaryDomain(n_bits)
    walsh = WalshTransform(n_bits)

    base_truth_table = generate_alternate_balanced_binary_vector(domain.space_cardinality())
    base_truth_table, _ = truth_tables_ea(n_bits, pop_size, warm_up, tournament_size, seed, verbose=False)

    generate = lambda: random_program(domain.space_cardinality(), min_length, max_length, rng, rand)
    evaluate = lambda x: walsh.granular_non_linearity(walsh.apply(execute_program(x, base_truth_table))[0])
    select = lambda x, y: tournament(x, y, tournament_size, rand)
    mate = lambda x, y: homologous_crossover(x, y, min_length, max_length, rand)
    mutate = lambda x: mutate_program(x, domain.space_cardinality(), min_length, max_length, rand)

    best_solution, best_score = evolutionary_algorithm(
        pop_size=pop_size,
        n_iter=n_iter - warm_up,
        generate=generate,
        evaluate=evaluate,
        select=select,
        mate=mate,
        mutate=mutate,
        rng=rng,
        rand=rand,
        verbose=verbose,
        cx_rate=0.5,
        mut_rate=0.5,
        mutually_exclusive=True,
        plateau_iter=1000000
    )

    return best_solution, best_score
