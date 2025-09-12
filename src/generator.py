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


# ===========================================================================
# Fixed Truth Tables
# ===========================================================================

def generate_alternate_balanced_binary_vector(size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.int8)
    mask = np.arange(size) % 2 == 0
    arr[mask] = 1
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
        start = rand.randint(0, N - max_interval_size)
        end = rand.randint(start + 1, start + max_interval_size - 1)
        return start, end
    else:
        raise ValueError(f"Unknown terminal type {kind}")


# === PRIMITIVE REGISTRY ===

def primitives():
    primitives_dict = {
        "swap": {
            "func": swap,
            "params": ["index", "index"]
        },
        #"consecutive_swaps": {
        #    "func": consecutive_swaps,
        #    "params": ["indexes"]
        #},
        #"reverse": {
        #    "func": reverse,
        #    "params": ["interval"]
        #},
        #"scramble": {
        #    "func": scramble,
        #    "params": ["interval", "seed"]
        #},
        #"rotate": {
        #    "func": rotate,
        #    "params": ["interval", "int"]
        #},
        #"shift_1s": {
        #    "func": shift_1s,
        #    "params": ["interval"]
        #}
    }

    return primitives_dict


# === PROGRAM MANAGEMENT ===

def clone_program(program: list[tuple[str, list]]) -> list[tuple[str, list]]:
    """Deep copy a program."""
    return [(a, [e for e in b]) for a, b in program]


def random_program(N: int, min_length: int, max_length: int, rng: np.random.Generator, rand: random.Random) -> list[tuple[str, list]]:
    """Generate a random permutation program (list of steps)."""
    primitives_dict = primitives()
    primitives_keys = sorted(list(primitives_dict.keys()))
    length = rng.integers(min_length, max_length + 1)
    program = []
    for _ in range(length):
        name = rand.choice(primitives_keys)
        entry = primitives_dict[name]
        params = [random_terminal(N, kind, rand) for kind in entry["params"]]
        program.append((name, params))
    return program


def execute_program(program: list[tuple[str, list]], L: np.ndarray) -> np.ndarray:
    """Execute program on array L (modifies in-place)."""
    truth_table = L.copy()
    primitives_dict = primitives()
    for (name, params) in program:
        func = primitives_dict[name]["func"]
        truth_table = func(*params, truth_table)
    return truth_table
