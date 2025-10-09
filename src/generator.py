import random
import numpy as np
from typing import Any


class Individual:
    def __init__(self, genome: Any, fitness: float, spectrum: np.ndarray, spectral_radius: int):
        self.genome = genome
        self.fitness = fitness
        self.spectrum = spectrum
        self.spectral_radius = spectral_radius


class IndividualProgram:
    def __init__(self, program: list[tuple[str, list]], base_genome: Any, genome: Any, fitness: float, spectrum: np.ndarray, spectral_radius: int):
        self.program = program
        self.genome = genome
        self.base_genome = base_genome
        self.fitness = fitness
        self.spectrum = spectrum
        self.spectral_radius = spectral_radius


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
        best_idx = -1
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


def block_swaps(pair_of_intervals: tuple[tuple[int, int], tuple[int, int]], L: np.ndarray) -> np.ndarray:
    interval1 = pair_of_intervals[0]
    interval2 = pair_of_intervals[1]
    start1, end1 = interval1[0], interval1[1]
    start2, end2 = interval2[0], interval2[1]
    start1, end1 = min(start1, end1), max(start1, end1)
    start2, end2 = min(start2, end2), max(start2, end2)
    len1 = end1 - start1 + 1
    len2 = end2 - start2 + 1
    if len1 != len2:
        raise ValueError("Intervals must be of the same length for block swap.")
    L[start1:end1 + 1], L[start2:end2 + 1] = L[start2:end2 + 1].copy(), L[start1:end1 + 1].copy()
    return L


def reverse(interval: tuple[int, int], L: np.ndarray) -> np.ndarray:
    start, end = interval[0], interval[1]
    start, end = min(start, end), max(start, end)
    L[start:end + 1] = L[start:end + 1][::-1]
    return L


def rotate(interval: tuple[int, int, int], L: np.ndarray) -> np.ndarray:
    start, end, k = interval[0], interval[1], interval[2]
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


def block_exchange(pair_of_intervals: tuple[tuple[int, int], tuple[int, int]], L: np.ndarray) -> np.ndarray:
    interval1 = pair_of_intervals[0]
    interval2 = pair_of_intervals[1]
    start1, end1 = interval1[0], interval1[1]
    start2, end2 = interval2[0], interval2[1]
    start1, end1 = min(start1, end1), max(start1, end1)
    start2, end2 = min(start2, end2), max(start2, end2)
    len1 = end1 - start1 + 1
    len2 = end2 - start2 + 1
    # This must simply exchange the two blocks, but the two blocks can be of different sizes.
    # Therefore, the new truth table will have the same size, but the two blocks will be swapped.
    # The area between the two blocks will be traslated accordingly.
    # For example, if we have:
    # L = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # and we swap intervals (1, 4) and (6, 8), we get:
    # L = [0, 6, 7, 8, 5, 1, 2, 3, 4, 9]
    # The area between the two blocks (5) is translated accordingly.
    if start1 > start2:
        # ensure start1 < start2
        start1, end1, start2, end2 = start2, end2, start1, end1
        len1, len2 = len2, len1
    middle = L[end1 + 1:start2].copy() if end1 + 1 < start2 else np.array([], dtype=L.dtype)
    segment1 = L[start1:end1 + 1].copy()
    segment2 = L[start2:end2 + 1].copy()
    L[start1:start1 + len2] = segment2
    L[start1 + len2:start1 + len2 + middle.shape[0]] = middle
    L[start1 + len2 + middle.shape[0]:start1 + len2 + middle.shape[0] + len1] = segment1
    return L


def consecutive_swaps(indexes: tuple[int, ...], L: np.ndarray) -> np.ndarray:
    for ii in range(0, len(indexes), 2):
        i, j = indexes[ii], indexes[ii + 1]
        L[i], L[j] = L[j], L[i]
    return L


def scramble(interval: tuple[int, int], seed: int, L: np.ndarray) -> np.ndarray:
    start, end = interval[0], interval[1]
    start, end = min(start, end), max(start, end)
    rng = np.random.default_rng(seed)
    sub = L[start:end + 1].copy()
    rng.shuffle(sub)
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
    elif kind == "interval_with_int":
        max_interval_size = N // 4
        if max_interval_size < 2:
            max_interval_size = 2
        start = rand.randint(0, N - max_interval_size)
        end = rand.randint(start + 1, start + max_interval_size - 1)
        k = rand.randint(1, end - start)  # rotate by at least 1 and at most interval size - 1
        return (start, end, k)
    elif kind == "small_interval":
        max_interval_size = N // 10
        if max_interval_size < 2:
            max_interval_size = 2
        start = rand.randint(0, N - max_interval_size)
        end = rand.randint(start + 1, start + max_interval_size - 1)
        return start, end
    elif kind == "pair_of_intervals":
        # must craft two intervals that do not overlap and with the same length
        max_interval_size = N // 8
        if max_interval_size < 2:
            max_interval_size = 2
        interval_size = rand.randint(2, max_interval_size)
        start1 = rand.randint(0, N - 2 * interval_size - 1)
        end1 = start1 + interval_size - 1
        start2 = rand.randint(end1 + 1, N - interval_size)
        end2 = start2 + interval_size - 1
        return (start1, end1), (start2, end2)
    elif kind == "pair_of_intervals_different_sizes":
        # must craft two intervals that do not overlap and with different lengths
        max_interval_size = N // 8
        if max_interval_size < 2:
            max_interval_size = 2
        interval_size1 = rand.randint(2, max_interval_size)
        interval_size2 = rand.randint(2, max_interval_size)
        start1 = rand.randint(0, N - interval_size1 - interval_size2 - 1)
        end1 = start1 + interval_size1 - 1
        start2 = rand.randint(end1 + 1, N - interval_size2)
        end2 = start2 + interval_size2 - 1
        return (start1, end1), (start2, end2)
    else:
        raise ValueError(f"Unknown terminal type {kind}")


# === PRIMITIVE REGISTRY ===

def primitives():
    primitives_dict = {
        "1-swap": {
            "func": swap,
            "params": ["index", "index"]
        },
        "2-block_swaps": {
           "func": block_swaps,
           "params": ["pair_of_intervals"]
        },
        "3-reverse": {
           "func": reverse,
           "params": ["interval"]
        },
        "4-rotate": {
           "func": rotate,
           "params": ["interval_with_int"]
        },
        "5-shift_1s": {
           "func": shift_1s,
           "params": ["interval"]
        },
        "6-block_exchange": {
           "func": block_exchange,
           "params": ["pair_of_intervals_different_sizes"]
        },
        # "7-consecutive_swaps": {
        #    "func": consecutive_swaps,
        #    "params": ["indexes"]
        # },
        # "8-scramble": {
        #    "func": scramble,
        #    "params": ["interval", "seed"]
        # },
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
