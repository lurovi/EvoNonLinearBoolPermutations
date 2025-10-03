import numpy as np
import random
from generator import clone_program


# ===========================================================================
# GA Truth Table Permutations Crossover
# ===========================================================================

def order_crossover(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Frequency-aware Order Crossover (OX) adapted for binary strings.
    Preserves 1/0 counts exactly.
    """
    size = len(parent1)
    cx1, cx2 = np.sort(rng.choice(size, 2, replace=False))

    child1, child2 = np.full(size, -1, dtype=np.int8), np.full(size, -1, dtype=np.int8)
    child1[cx1:cx2] = parent1[cx1:cx2].copy()
    child2[cx1:cx2] = parent2[cx1:cx2].copy()

    def fill(child, donor, taken):
        count1 = np.sum(child == 1)
        count0 = np.sum(child == 0)
        max1 = np.sum(donor == 1)
        max0 = len(donor) - max1
        for i in range(size):
            if child[i] == -1:
                val = donor[taken % size]
                taken += 1
                if val == 1 and count1 < max1:
                    child[i] = 1
                    count1 += 1
                elif val == 0 and count0 < max0:
                    child[i] = 0
                    count0 += 1
                else:
                    # put the opposite to maintain balance
                    child[i] = 1 - val
        return child

    child1 = fill(child1, parent2, cx2).astype(np.int8)
    child2 = fill(child2, parent1, cx2).astype(np.int8)
    child = child1 if rng.random() < 0.5 else child2
    return child


def uniform_crossover_with_repair(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Uniform crossover + repair to enforce balanced counts.
    """
    size = len(parent1)
    mask = rng.integers(0, 2, size=size)
    child = np.where(mask, parent1.copy(), parent2.copy())

    # repair if counts don't match
    n1 = np.sum(parent1)
    excess1 = int(np.sum(child) - n1)
    if excess1 > 0:
        idx = rng.choice(np.where(child == 1)[0], size=excess1, replace=False)
        child[idx] = 0
    elif excess1 < 0:
        idx = rng.choice(np.where(child == 0)[0], size=-excess1, replace=False)
        child[idx] = 1
    return child.astype(np.int8)


def position_based_crossover(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Position-based crossover with count repair.
    """
    size = len(parent1)
    positions = rng.choice(size, size=size // 2, replace=False)
    child = np.full(size, -1, dtype=np.int8)
    child[positions] = parent1[positions].copy()

    # fill from parent2
    n1 = np.sum(parent1)
    mask = child == -1
    child[mask] = parent2[mask].copy()

    # repair
    excess1 = int(np.sum(child) - n1)
    if excess1 > 0:
        idx = rng.choice(np.where(child == 1)[0], size=excess1, replace=False)
        child[idx] = 0
    elif excess1 < 0:
        idx = rng.choice(np.where(child == 0)[0], size=-excess1, replace=False)
        child[idx] = 1
    return child.astype(np.int8)


def cycle_crossover_binary(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Cycle Crossover (CX) adapted for binary strings with duplicates.
    Ensures exact preservation of 0/1 counts.
    """
    p1 = np.asarray(parent1.copy(), dtype=np.int8)
    p2 = np.asarray(parent2.copy(), dtype=np.int8)
    n = p1.size
    if n != p2.size:
        raise ValueError("Parents must have the same length.")
    if int(p1.sum()) != int(p2.sum()):
        raise ValueError("Parents must have the same number of ones.")

    c1 = int(p1.sum())
    c0 = n - c1

    # ---- Tokenize: map each 0/1 to a unique token using occurrence rank
    # Tokens are contiguous: zeros -> [0 .. c0-1], ones -> [c0 .. n-1]
    def tokenize(seq: np.ndarray, c0_: int) -> np.ndarray:
        out = np.empty(seq.size, dtype=np.int64)
        next0, next1 = 0, c0_
        for ii, b in enumerate(seq):
            if b == 0:
                out[ii] = next0
                next0 += 1
            else:
                out[ii] = next1
                next1 += 1
        return out

    A = tokenize(p1, c0)  # tokens for parent1
    B = tokenize(p2, c0)  # tokens for parent2 (same token set, different order)

    # ---- Classic Cycle Crossover on unique-token permutations A, B
    invA = np.empty(n, dtype=np.int64)
    invA[A] = np.arange(n)  # inverse permutation: token -> index in A

    visited = np.zeros(n, dtype=bool)
    child1_tokens = np.empty(n, dtype=np.int64)
    child2_tokens = np.empty(n, dtype=np.int64)

    use_from_A = True  # alternate cycles
    for start in range(n):
        if visited[start]:
            continue
        i = start
        cycle_indexes = []
        while not visited[i]:
            visited[i] = True
            cycle_indexes.append(i)
            # follow cycle: position i in B has token B[i]; find where that token sits in A
            i = invA[B[i]]

        indexes = np.asarray(cycle_indexes, dtype=np.int64)
        if use_from_A:
            child1_tokens[indexes] = A[indexes]
            child2_tokens[indexes] = B[indexes]
        else:
            child1_tokens[indexes] = B[indexes]
            child2_tokens[indexes] = A[indexes]
        use_from_A = not use_from_A

    # ---- Detokenize: tokens < c0 -> 0, else 1
    child1 = (child1_tokens >= c0).astype(np.int8)
    child2 = (child2_tokens >= c0).astype(np.int8)
    child = child1 if rng.random() < 0.5 else child2
    return child


def pmx_binary(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Partially Matched Crossover (PMX) adapted for binary strings with duplicates.
    Ensures preservation of the 0/1 counts.
    """
    p1 = np.asarray(parent1.copy(), dtype=np.int8)
    p2 = np.asarray(parent2.copy(), dtype=np.int8)
    n = p1.size
    if n != p2.size:
        raise ValueError("Parents must have the same length.")
    if int(p1.sum()) != int(p2.sum()):
        raise ValueError("Parents must have the same number of ones.")

    c1 = int(p1.sum())
    c0 = n - c1

    # ---- Tokenize: map each 0/1 to a unique token using occurrence rank
    def tokenize(seq: np.ndarray, c0_: int) -> np.ndarray:
        out = np.empty(seq.size, dtype=np.int64)
        next0, next1 = 0, c0_
        for i, b in enumerate(seq):
            if b == 0:
                out[i] = next0
                next0 += 1
            else:
                out[i] = next1
                next1 += 1
        return out

    A = tokenize(p1, c0)  # tokens for parent1
    B = tokenize(p2, c0)  # tokens for parent2

    # ---- PMX on unique tokens
    child1_tokens = np.full(n, -1, dtype=np.int64)
    child2_tokens = np.full(n, -1, dtype=np.int64)

    # Pick crossover points
    pt1, pt2 = sorted(rng.choice(n, size=2, replace=False))

    # Copy slice from parents
    child1_tokens[pt1:pt2] = A[pt1:pt2]
    child2_tokens[pt1:pt2] = B[pt1:pt2]

    # Build mapping for swaps inside the slice
    mapping1 = dict(zip(A[pt1:pt2], B[pt1:pt2]))
    mapping2 = dict(zip(B[pt1:pt2], A[pt1:pt2]))

    def pmx_fill(child_tokens, parent_tokens, mapping):
        for i in range(n):
            if pt1 <= i < pt2:
                continue
            token = parent_tokens[i]
            while token in mapping and mapping[token] != token:
                token = mapping[token]
            child_tokens[i] = token

    pmx_fill(child1_tokens, B, mapping2)  # fill from other parent
    pmx_fill(child2_tokens, A, mapping1)

    # ---- Detokenize: tokens < c0 -> 0, else 1
    child1 = (child1_tokens >= c0).astype(np.int8)
    child2 = (child2_tokens >= c0).astype(np.int8)
    child = child1 if rng.random() < 0.5 else child2
    return child


# ===========================================================================
# Linear GP Program Permutations Crossover
# ===========================================================================

# Program type alias
Program = list[tuple[str, list]]


def enforce_length(prog: Program, min_length: int, max_length: int, rand: random.Random) -> Program:
    """
    Ensure program length is within [min_length, max_length].
    - If too short: pad with copies of random steps from prog.
    - If too long: truncate randomly.
    """
    if len(prog) < min_length:
        # Pad by duplicating random steps
        while len(prog) < min_length:
            prog.append(clone_program([rand.choice(prog)])[0])
    elif len(prog) > max_length:
        # Truncate randomly until valid length
        while len(prog) > max_length:
            del prog[rand.randrange(len(prog))]
    return prog


def one_point_crossover(p1: Program, p2: Program, min_length: int, max_length: int, rand: random.Random) -> Program:
    """
    One-point crossover between two programs.
    """
    if len(p1) < 2 or len(p2) < 2:
        child = clone_program(p1) if len(p2) < 2 else clone_program(p2)
        return enforce_length(child, min_length, max_length, rand)

    p1, p2 = clone_program(p1), clone_program(p2)

    cx_point1 = rand.randint(1, len(p1) - 1)
    cx_point2 = rand.randint(1, len(p2) - 1)

    c1 = p1[:cx_point1] + p2[cx_point2:]
    c2 = p2[:cx_point2] + p1[cx_point1:]

    child = c1 if rand.random() < 0.5 else c2
    return enforce_length(child, min_length, max_length, rand)


def two_point_crossover(p1: Program, p2: Program, min_length: int, max_length: int, rand: random.Random) -> Program:
    """
    Two-point crossover between two programs.
    """
    if len(p1) < 3 or len(p2) < 3:
        return one_point_crossover(p1, p2, min_length, max_length, rand)

    p1, p2 = clone_program(p1), clone_program(p2)

    a1, a2 = sorted(rand.sample(range(len(p1)), 2))
    b1, b2 = sorted(rand.sample(range(len(p2)), 2))

    c1 = p1[:a1] + p2[b1:b2] + p1[a2:]
    c2 = p2[:b1] + p1[a1:a2] + p2[b2:]

    child = c1 if rand.random() < 0.5 else c2
    return enforce_length(child, min_length, max_length, rand)


def uniform_step_crossover(p1: Program, p2: Program, min_length: int, max_length: int, rand: random.Random) -> Program:
    """
    Uniform step crossover: child is built step-by-step from p1 and p2.
    Ensures length is in [min_length, max_length].
    """
    p1, p2 = clone_program(p1), clone_program(p2)

    # Target length between min and max
    target_len = rand.randint(min_length, max_length)

    c = []
    for i in range(target_len):
        if i < len(p1) and i < len(p2):
            step = p1[i] if rand.random() < 0.5 else p2[i]
        elif i < len(p1):
            step = p1[i]
        elif i < len(p2):
            step = p2[i]
        else:
            # If both shorter, randomly reuse last available
            step = rand.choice(p1 + p2)
        c.append(step)

    return c


def homologous_crossover(p1: Program, p2: Program, min_length: int, max_length: int, rand: random.Random) -> Program:
    """
    Homologous crossover for linear GP programs.
    Aligns the two parents and swaps corresponding subsequences.
    Ensures child length is in [min_length, max_length].
    """

    if len(p1) < 2 or len(p2) < 2:
        # fallback to uniform crossover
        return uniform_step_crossover(p1, p2, min_length, max_length, rand)

    p1, p2 = clone_program(p1), clone_program(p2)

    # Align lengths
    min_len = min(len(p1), len(p2))
    max_len = max(len(p1), len(p2))

    # Select aligned segment
    i, j = sorted(rand.sample(range(min_len), 2))

    # Child starts as a copy of parent1
    c = clone_program(p1)

    # Replace aligned segment with p2's segment
    c[i:j] = clone_program(p2[i:j])

    # Enforce length bounds
    if len(c) < min_length:
        # pad with steps from p2 (cycling if necessary)
        while len(c) < min_length:
            c.append(clone_program([rand.choice(p1 + p2)])[0])
    elif len(c) > max_length:
        # truncate
        c = c[:max_length]

    return c
