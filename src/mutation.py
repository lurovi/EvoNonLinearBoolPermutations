import numpy as np
import random
from collections.abc import Callable
from generator import execute_program, initialize_population_programs, primitives, random_program, random_terminal, clone_program
from full_binary_domain import FullBinaryDomain
from algorithm import evolutionary_algorithm_programs, simulated_annealing_truth_tables, random_search_truth_tables
from crossover import homologous_crossover
from walsh_transform import WalshTransform


# ===========================================================================
# GA Truth Table Permutations Mutation
# ===========================================================================

def swap_mutation(ind: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_ind = ind.copy()
    i, j = rng.choice(len(n_ind), 2, replace=False)
    n_ind[i], n_ind[j] = n_ind[j], n_ind[i]
    return n_ind

def consecutive_swap_mutation(ind: np.ndarray, k1: int, k2: int, rng: np.random.Generator) -> np.ndarray:
    n_ind = ind.copy()
    k = rng.integers(k1, k2 + 1)
    for _ in range(k):
        i, j = rng.choice(len(n_ind), 2, replace=False)
        n_ind[i], n_ind[j] = n_ind[j], n_ind[i]
    return n_ind

def inversion_mutation(ind: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_ind = ind.copy()
    i, j = np.sort(rng.choice(len(n_ind), 2, replace=False))
    n_ind[i:j] = n_ind[i:j][::-1]
    return n_ind


def scramble_mutation(ind: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_ind = ind.copy()
    i, j = np.sort(rng.choice(len(n_ind), 2, replace=False))
    slice_ = n_ind[i:j].copy()
    rng.shuffle(slice_)
    n_ind[i:j] = slice_
    return n_ind


def local_search_swaps(ind: np.ndarray, eval_fn: Callable[[np.ndarray], float], rand: random.Random, eval_budget: int) -> np.ndarray:
    """
    Try up to eval_budget swap proposals (swap 1<->0). Accept if improves fitness.
    Returns (best_individual, best_score, used_evals)
    """
    best = ind.copy()
    best_score = eval_fn(best)
    used = 1
    ones = np.where(best == 1)[0].tolist()
    zeros = np.where(best == 0)[0].tolist()

    for _ in range(eval_budget - 1):
        i = rand.choice(ones)
        j = rand.choice(zeros)
        # swap
        best[i], best[j] = best[j], best[i]
        score = eval_fn(best)
        used += 1
        if score > best_score:
            # accept and update lists
            best_score = score
            # update indices lists
            ones.remove(i); ones.append(j)
            zeros.remove(j); zeros.append(i)
        else:
            # revert
            best[i], best[j] = best[j], best[i]
    return best


def spectrum_aware_swap(
    ind: np.ndarray,
    walsh: WalshTransform,
    evaluate: Callable[[np.ndarray], float],
    rand: random.Random,
    eval_budget: int,
    top_frac: float = 0.1,
) -> np.ndarray:
    """
    Spectrum-aware swap mutation/local search.

    - Computes FWHT of the polar form.
    - Identifies top |coeff| Walsh spectrum indices.
    - Proposes swaps of 1<->0 at positions strongly influencing those indices.
    - Accepts only if improves fitness (hill-climbing).
    - Respects eval_budget (#fitness evaluations).

    Args:
        ind: numpy array (0/1), balanced.
        walsh: WalshTransform
        evaluate: fitness evaluator.
        rand: Python random.Random
        eval_budget: maximum number of fitness evaluations.
        top_frac: fraction of spectrum coefficients to consider (e.g., 0.1 = top 10%).

    Returns:
        best_ind
    """
    size = ind.size
    # initial fitness
    best = ind.copy()
    best_score = evaluate(best)
    used = 1

    # FWHT
    coeffs = walsh.apply(best)[0]

    # pick top |coeff| indices
    abs_coeffs = np.abs(coeffs)
    k_top = max(1, int(top_frac * size))
    top_idx = np.argpartition(abs_coeffs, -k_top)[-k_top:]

    # Precompute influence mask:
    # In Walsh spectrum, coefficient index c affects positions where (i & c) has odd parity
    # We'll build for each top coeff a list of candidate bit positions
    influence_positions = []
    for c in top_idx:
        mask = [i for i in range(size) if bin(i & c).count("1") % 2 == 1]
        influence_positions.append(mask)

    # propose swaps
    for _ in range(eval_budget - 1):
        # pick one top coefficient and its positions
        pos_list = influence_positions[rand.randrange(len(influence_positions))]
        if not pos_list:
            continue
        # pick one 1 and one 0 among those positions
        ones = [i for i in pos_list if best[i] == 1]
        zeros = [i for i in pos_list if best[i] == 0]
        if not ones or not zeros:
            continue
        i = rand.choice(ones)
        j = rand.choice(zeros)
        # swap
        best[i], best[j] = best[j], best[i]
        score = evaluate(best)
        used += 1
        if score > best_score:
            best_score = score
        else:
            # revert
            best[i], best[j] = best[j], best[i]
    return best


# def programs_random_search_enhancing_balanced_truth_table(
#         domain: FullBinaryDomain,
#         walsh: WalshTransform,
#         ind: np.ndarray,
#         eval_budget: int,
#         min_length: int,
#         max_length: int,
#         sampling_probabilities: list[float],
#         rng: np.random.Generator,
#         rand: random.Random,
# ) -> np.ndarray:
#     ind = ind.copy()
#     generate = lambda: random_program(domain.space_cardinality(), sampling_probabilities, min_length, max_length, rng, rand)
#     evaluate = lambda x: walsh.granular_non_linearity(walsh.apply(execute_program(x, ind))[0])

#     best_solution, _ = random_search_truth_tables(
#         n_iter=eval_budget,
#         generate=generate,
#         evaluate=evaluate,
#         verbose=False,
#     )

#     return execute_program(best_solution, ind)


# def programs_simulated_annealing_enhancing_balanced_truth_table(
#         domain: FullBinaryDomain,
#         walsh: WalshTransform,
#         ind: np.ndarray,
#         eval_budget: int,
#         min_length: int,
#         max_length: int,
#         sampling_probabilities: list[float],
#         rng: np.random.Generator,
#         rand: random.Random,
# ) -> np.ndarray:
#     ind = ind.copy()
#     generate = lambda: random_program(domain.space_cardinality(), sampling_probabilities, min_length, max_length, rng, rand)
#     evaluate = lambda x: walsh.granular_non_linearity(walsh.apply(execute_program(x, ind))[0])
#     mutate = lambda x: mutate_program(x, domain.space_cardinality(), sampling_probabilities, min_length, max_length, rand)

#     best_solution, _ = simulated_annealing(
#         n_iter=eval_budget,
#         generate=generate,
#         evaluate=evaluate,
#         mutate=mutate,
#         rand=rand,
#         verbose=False,
#     )

#     return execute_program(best_solution, ind)


# def programs_evolutionary_algorithm_enhancing_balanced_truth_table(
#         domain: FullBinaryDomain,
#         walsh: WalshTransform,
#         ind: np.ndarray,
#         pop_size: int,
#         n_iter: int,
#         min_length: int,
#         max_length: int,
#         sampling_probabilities: list[float],
#         rng: np.random.Generator,
#         rand: random.Random,
# ) -> np.ndarray:
#     ind = ind.copy()
#     initialize = lambda x: initialize_population_programs(x, domain.space_cardinality(), sampling_probabilities, min_length, max_length, rng, rand)
#     evaluate = lambda x: walsh.granular_non_linearity(walsh.apply(execute_program(x, ind))[0])
#     mate = lambda x, y: homologous_crossover(x, y, min_length, max_length, rand)
#     mutate = lambda x: mutate_program(x, domain.space_cardinality(), sampling_probabilities, min_length, max_length, rand)

#     best_solution, _ = evolutionary_algorithm_programs(
#         pop_size=pop_size,
#         n_iter=n_iter,
#         initialize=initialize,
#         evaluate=evaluate,
#         mate=mate,
#         mutate=mutate,
#         rng=rng,
#         rand=rand,
#         verbose=False,
#         cx_rate=0.5,
#         mut_rate=0.5,
#         mutually_exclusive=True,
#         plateau_iter=1000000,
#         # Cellular GA parameters
#         pressure=2,
#         torus_dim=0,
#         radius=0,
#         pop_shape=tuple(),
#         cmp_rate=0.0,

#     )

#     return execute_program(best_solution.genome, ind)

# ===========================================================================
# Linear GP Program Permutations Mutation
# ===========================================================================

# Program type alias
Program = list[tuple[str, list]]


def mutate_program(
    program: Program,
    N: int,
    sampling_probabilities: list[float],
    min_length: int,
    max_length: int,
    rand: random.Random,
) -> Program:
    """
    Mutate a program (list of steps), enforcing length constraints.

    - Insert: add a random new step if len < max_length.
    - Delete: remove a step if len > min_length.
    - Modify: change primitive or parameters of a step.
    """
    prog = clone_program(program)
    op = rand.choice(["INSERT", "DELETE", "MODIFY"])

    primitives_dict = primitives()
    primitives_keys = sorted(list(primitives_dict.keys()))

    # --- INSERT ---
    if op == "INSERT" and len(prog) < max_length:
        new_name = rand.choices(primitives_keys, weights=sampling_probabilities, k=1)[0]
        new_params = [random_terminal(N, kind, rand) for kind in primitives_dict[new_name]["params"]]
        insert_pos = rand.randrange(len(prog) + 1)
        prog.insert(insert_pos, (new_name, new_params))

    # --- DELETE ---
    elif op == "DELETE" and len(prog) > min_length:
        del_pos = rand.randrange(len(prog))
        del prog[del_pos]

    # --- MODIFY ---
    else:
        if len(prog) > 0:

            idx = rand.randrange(len(prog))
            name, params = prog[idx]

            if rand.random() < 0.5:
                # Replace entire primitive
                new_name = rand.choices(primitives_keys, weights=sampling_probabilities, k=1)[0]
                new_params = [random_terminal(N, kind, rand) for kind in primitives_dict[new_name]["params"]]
                prog[idx] = (new_name, new_params)
            else:
                # Mutate one parameter
                if len(params) > 0:
                    p_idx = rand.randrange(len(params))
                    kind = primitives_dict[name]["params"][p_idx]
                    params[p_idx] = random_terminal(N, kind, rand)
                    prog[idx] = (name, params)

    return prog
