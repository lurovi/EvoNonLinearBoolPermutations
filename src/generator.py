import random
import numpy as np


# ===========================================================================
# Truth Table Representations
# ===========================================================================

def generate_random_binary_vector(size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(low=0, high=2, size=size, dtype=np.int8)


# ===========================================================================
# Balanced Truth Table Representations
# ===========================================================================

def generate_random_balanced_binary_vector(size: int, rng: np.random.Generator) -> np.ndarray:
    arr = np.zeros(size, dtype=np.int8)
    mask = np.arange(size) % 2 == 0
    arr[mask] = 1
    rng.shuffle(arr)
    return arr


def uniform_initial_population(size: int, pop_size: int, rng: np.random.Generator, rand: random.Random) -> list[np.ndarray]:
    """
    Uniform random init: just sample uniformly at random from the space.
    Generates balanced candidates by random shuffle of base balanced vector.
    """
    base = np.zeros(size, dtype=np.int8)
    mask = np.arange(size) % 2 == 0
    base[mask] = 1
    population = []
    for _ in range(pop_size):
        individual = base.copy()
        rng.shuffle(individual)
        population.append(individual)
    return population


def greedy_maxmin_initial_population(size: int, pop_size: int, rng: np.random.Generator, rand: random.Random) -> list[np.ndarray]:
    """
    Greedy max-min diversity init: start from random, then iteratively add the
    candidate that maximizes min Hamming distance to current set.
    Generates balanced candidates by random shuffle of base balanced vector.
    """
    base = np.zeros(size, dtype=np.int8)
    base[:size//2] = 1
    pool_size = pop_size * 10
    pool = []
    for _ in range(pool_size):
        tmp = base.copy()
        rng.shuffle(tmp)
        pool.append(tmp)
    # pick one random first
    selected = [pool.pop(rng.integers(len(pool)))]
    while len(selected) < pop_size:
        best_idx = None
        best_min = -1
        for i, cand in enumerate(pool):
            dists = [np.sum(cand != s) for s in selected]
            min_dist = min(dists)
            if min_dist > best_min:
                best_min = min_dist
                best_idx = i
        selected.append(pool.pop(best_idx))
    return selected


# ===========================================================================
# Fixed Known Truth Tables
# ===========================================================================

def generate_alternate_balanced_binary_vector_one_zero(size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.int8)
    mask = np.arange(size) % 2 == 0
    arr[mask] = 1
    return arr

def generate_alternate_balanced_binary_vector_zero_one(size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.int8)
    mask = np.arange(size) % 2 == 1
    arr[mask] = 1
    return arr


def generate_half_ones_half_zeros_binary_vector(size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.int8)
    half_size = size // 2
    arr[:half_size] = 1
    return arr


def generate_half_zeros_half_ones_binary_vector(size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.int8)
    half_size = size // 2
    arr[half_size:] = 1
    return arr


def generate_quarter_ones_half_zeros_quarter_ones_binary_vector(size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.int8)
    quarter_size = size // 4
    arr[:quarter_size] = 1
    arr[3*quarter_size:] = 1
    return arr

def generate_quarter_zeros_half_ones_quarter_zeros_binary_vector(size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.int8)
    quarter_size = size // 4
    arr[quarter_size:3*quarter_size] = 1
    return arr


def generate_quarter_ones_quarter_zeros_half_ones_binary_vector(size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.int8)
    quarter_size = size // 4
    arr[:quarter_size] = 1
    arr[2*quarter_size:3*quarter_size] = 1
    return arr


def generate_quarter_zeros_quarter_ones_half_zeros_binary_vector(size: int) -> np.ndarray:
    arr = np.ones(size, dtype=np.int8)
    quarter_size = size // 4
    arr[:quarter_size] = 0
    arr[2*quarter_size:3*quarter_size] = 0
    return arr


def generate_eighths_alternating_binary_vector(size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.int8)
    eighth_size = size // 8
    for i in range(0, 8, 2):
        arr[i*eighth_size:(i+1)*eighth_size] = 1
    return arr


def generate_eighths_alternating_binary_vector_starting_with_zero(size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.int8)
    eighth_size = size // 8
    for i in range(1, 8, 2):
        arr[i*eighth_size:(i+1)*eighth_size] = 1
    return arr


# ===========================================================================
# Permutation Programs
# ===========================================================================

# === PRIMITIVES ===

def swap(i: int, j: int, L: np.ndarray) -> np.ndarray:
    L[i], L[j] = L[j], L[i]
    return L


def consecutive_swaps(indexes: tuple[int, ...], L: np.ndarray) -> np.ndarray:
    for ii in range(0, len(indexes), 2):
        i, j = indexes[ii], indexes[ii + 1]
        L[i], L[j] = L[j], L[i]
    return L


def block_swaps(indexes: tuple[int, ...], L: np.ndarray) -> np.ndarray:
    half = len(indexes) // 2
    sorted_indexes = sorted(indexes)
    for ii in range(half):
        i, j = sorted_indexes[ii], sorted_indexes[ii + half]
        L[i], L[j] = L[j], L[i]
    return L


def reverse(interval: tuple[int, int], L: np.ndarray) -> np.ndarray:
    start, end = interval[0], interval[1]
    start, end = min(start, end), max(start, end)
    L[start:end + 1] = L[start:end + 1][::-1]
    return L


def scramble(interval: tuple[int, int], seed: int, L: np.ndarray) -> np.ndarray:
    start, end = interval[0], interval[1]
    start, end = min(start, end), max(start, end)
    rng = np.random.default_rng(seed)
    sub = L[start:end + 1].copy()
    rng.shuffle(sub)
    L[start:end + 1] = sub
    return L


def rotate(interval: tuple[int, int], k: int, L: np.ndarray) -> np.ndarray:
    start, end = interval[0], interval[1]
    start, end = min(start, end), max(start, end)
    sub = L[start:end + 1].copy()
    n = sub.shape[0]
    k %= n
    if k > 0:
        L[start:end + 1] = np.concatenate((sub[k:], sub[:k]))
    return L


def shift_1s(interval: tuple[int, int], L: np.ndarray) -> np.ndarray:
    """Shift all 1s forward by one position (cyclic)."""
    start, end = interval[0], interval[1]
    start, end = min(start, end), max(start, end)
    sub = L[start:end + 1].copy()
    n = sub.shape[0]
    ones = np.where(sub == 1)[0]
    for i in reversed(ones):
        j = (i + 1) % n
        sub[i], sub[j] = sub[j], sub[i]
    L[start:end + 1] = sub
    return L


# === TERMINALS ===

def random_terminal(N: int, kind: str, rand: random.Random):
    """Generate a random terminal of the given kind."""
    if kind == "index":
        return rand.randint(0, N - 1)
    elif kind == "indexes":
        size = rand.randint(2, N // 2)
        if size % 2 != 0:
            size += 1
        return tuple([rand.randint(0, N - 1) for _ in range(size)])
    elif kind == "seed":
        return rand.randint(1, 99999)
    elif kind == "int":
        return rand.randint(1, N - 1)
    elif kind == "interval":
        max_interval_size = N // 4
        if max_interval_size < 2:
            max_interval_size = 2
        start = rand.randint(0, N - max_interval_size)
        end = rand.randint(start + 1, start + max_interval_size - 1)
        return start, end
    elif kind == "small_interval":
        max_interval_size = N // 10
        if max_interval_size < 2:
            max_interval_size = 2
        start = rand.randint(0, N - max_interval_size)
        end = rand.randint(start + 1, start + max_interval_size - 1)
        return start, end
    else:
        raise ValueError(f"Unknown terminal type {kind}")


# === PRIMITIVE REGISTRY ===

def primitives():
    primitives_dict = {
        "1-swap": {
            "func": swap,
            "params": ["index", "index"]
        },
        "2-consecutive_swaps": {
           "func": consecutive_swaps,
           "params": ["indexes"]
        },
        "3-block_swaps": {
           "func": block_swaps,
           "params": ["indexes"]
        },
        "4-reverse": {
           "func": reverse,
           "params": ["interval"]
        },
        "5-rotate": {
           "func": rotate,
           "params": ["interval", "int"]
        },
        "6-shift_1s": {
           "func": shift_1s,
           "params": ["interval"]
        },
        "7-scramble": {
           "func": scramble,
           "params": ["interval", "seed"]
        },
    }

    return primitives_dict


# === PROGRAM MANAGEMENT ===

def clone_program(program: list[tuple[str, list]]) -> list[tuple[str, list]]:
    """Deep copy a program."""
    return [(a, [e for e in b]) for a, b in program]


def random_program(N: int, sampling_probabilities: list[float], min_length: int, max_length: int, rng: np.random.Generator, rand: random.Random) -> list[tuple[str, list]]:
    """Generate a random permutation program (list of steps)."""
    primitives_dict = primitives()
    primitives_keys = sorted(list(primitives_dict.keys()))
    length = int(rng.integers(min_length, max_length + 1))
    program = []
    names = rand.choices(primitives_keys, weights=sampling_probabilities, k=length)
    for name in names:
        entry = primitives_dict[name]
        params = [random_terminal(N, kind, rand) for kind in entry["params"]]
        program.append((name, params))
    return program


def initialize_population_programs(pop_size: int, N: int, sampling_probabilities: list[float], min_length: int, max_length: int, rng: np.random.Generator, rand: random.Random) -> list[list[tuple[str, list]]]:
    """Initialize a population of random permutation programs."""
    return [random_program(N, sampling_probabilities, min_length, max_length, rng, rand) for _ in range(pop_size)]


def execute_program(program: list[tuple[str, list]], L: np.ndarray) -> np.ndarray:
    """Execute program on array L (modifies in-place)."""
    truth_table = L.copy()
    primitives_dict = primitives()
    for (name, params) in program:
        func = primitives_dict[name]["func"]
        truth_table = func(*params, truth_table)
    return truth_table
