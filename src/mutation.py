import numpy as np
import random
from generator import primitives, random_terminal, clone_program


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


# ===========================================================================
# Linear GP Program Permutations Mutation
# ===========================================================================

# Program type alias
Program = list[tuple[str, list]]


def mutate_program(
    program: Program,
    n: int,
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
        new_params = [random_terminal(n, kind, rand) for kind in primitives_dict[new_name]["params"]]
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
                new_params = [random_terminal(n, kind, rand) for kind in primitives_dict[new_name]["params"]]
                prog[idx] = (new_name, new_params)
            else:
                # Mutate one parameter
                if len(params) > 0:
                    p_idx = rand.randrange(len(params))
                    kind = primitives_dict[name]["params"][p_idx]
                    params[p_idx] = random_terminal(n, kind, rand)
                    prog[idx] = (name, params)

    return prog
