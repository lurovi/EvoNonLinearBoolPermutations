from time import time
from algorithm import random_search, simulated_annealing, evolutionary_algorithm
from crossover import *
from mutation import *
from selection import *
from generator import *
from full_binary_domain import *
from method import *
from walsh_transform import *


if __name__ == "__main__":

    seed = 40

    rand = random.Random(seed)
    rng = np.random.default_rng(seed)

    n_bits = 4
    sampling_probabilities = [0.7, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0]

    domain = FullBinaryDomain(n_bits)
    walsh = WalshTransform(n_bits)
    N = domain.space_cardinality()
    base_truth_table = generate_alternate_balanced_binary_vector(N)

    generate = lambda: random_program(N, sampling_probabilities, 2, 10, rng, rand)
    evaluate = lambda x: walsh.granular_non_linearity(walsh.apply(execute_program(x, base_truth_table))[0])
    select = lambda x, y: tournament(x, y, 2, rand)
    mate = lambda x, y: homologous_crossover(x, y, 2, 10, rand)
    mutate = lambda x: mutate_program(x, N, sampling_probabilities, 2, 10, rand)

    pop = [generate() for _ in range(10)]
    fitness = []
    for program in pop:
        print(program)
        print()
        fitness.append(evaluate(program))
    print(fitness)
    pop = [select(pop, fitness) for _ in range(10)]
    fitness = []
    for program in pop:
        print(program)
        print()
        fitness.append(evaluate(program))
    print(fitness)
    
    print('MATING')
    print(pop[0])
    print(pop[1])
    print(mate(pop[0], pop[1]))
    print(pop[1])
    print(pop[2])
    print(mate(pop[1], pop[2]))
    print(pop[2])
    print(pop[3])
    print(mate(pop[2], pop[3]))
    print()
    print('MUTATING')
    print(pop[0])
    print(mutate(pop[0]))
    print(pop[1])
    print(mutate(pop[1]))
    print(pop[2])
    print(mutate(pop[2]))
    print()
    print('================================== FINAL TEST ================================')
    start_time = time()
    sampling_probabilities = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_program, best_score = truth_tables_ea(n_bits=10, pop_size=50, n_iter=500, tournament_size=5, seed=1, verbose=True)
    #best_program, best_score = programs_rs(n_bits=10, sampling_probabilities=sampling_probabilities, warm_up=50000, n_iter=500000, seed=1, min_length=2, max_length=40, verbose=True)
    #best_program, best_score = programs_sa(n_bits=10, sampling_probabilities=sampling_probabilities, warm_up=50000, n_iter=500000, seed=1, min_length=2, max_length=10, verbose=True)
    best_program, best_score = programs_ea(n_bits=10, sampling_probabilities=sampling_probabilities, pop_size=50, warm_up=100, n_iter=500, tournament_size=5, seed=1, min_length=2, max_length=20, verbose=True)
    end_time = time()
    print(f"Best program: {best_program}")
    print(f"Best score: {best_score}")
    print(f"Time taken: {(end_time - start_time) / 60} minutes")
