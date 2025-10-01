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

    seed = 42
    rng = np.random.default_rng(seed)
    rand = random.Random(seed)

    domain = FullBinaryDomain(5)
    walsh = WalshTransform(5)

    print(domain.covering_radius_bound())
    print(walsh.number_of_ones_for_each_number())

    print(generate_random_binary_vector(16, rng))
    print(generate_random_balanced_binary_vector(16, rng))
    print(generate_alternate_balanced_binary_vector_one_zero(16))
    print('==========Generation============')
    pop = [generate_random_balanced_binary_vector(16, rng) for _ in range(10)]
    scores = [walsh.granular_non_linearity(walsh.apply(pop[i])[0]) for i in range(len(pop))]
    for i in range(len(pop)):
        print(pop[i])
        print(scores[i])
        print()

    pop = [tournament(pop, scores, 4, rand) for i in range(len(pop))]
    print('========Selection=============')
    for i in range(len(pop)):
        print(pop[i])
        print(scores[i])
        print()
    print('=========Mutation==============')
    print(swap_mutation(pop[-1], rng))
    print(inversion_mutation(pop[-1], rng))
    print(scramble_mutation(pop[-1], rng))
    print('=========Crossover=============')
    print("Parent 1", pop[0])
    print("Parent 2", pop[5])
    parent1, parent2 = pop[0], pop[5]
    print(uniform_crossover_with_repair(parent1.copy(), parent2.copy(), rng))
    print(position_based_crossover(parent1.copy(), parent2.copy(), rng))

    print(order_crossover(parent1.copy(), parent2.copy(), rng))
    print(cycle_crossover_binary(parent1.copy(), parent2.copy()))
    print(pmx_binary(parent1.copy(), parent2.copy(), rng))
    print('=========Optimization=============')
    #truth_tables_rs(10, 50000, 11, True)
    #truth_tables_sa(10, 50000, 11, True)
    #truth_tables_ea(10, 50, 10000, 3, 11, True)
    #programs_rs(10, 5000, 50000, 11, 2, 20, True)
    #programs_sa(10, 1000, 100000, 11, 2, 20, True)
    #programs_ea(10, 50, 100, 10000, 2, 11, 2, 40, True)

    seed = 40

    rand = random.Random(seed)
    rng = np.random.default_rng(seed)

    n_bits = 4
    sampling_probabilities = [0.7, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0]

    domain = FullBinaryDomain(n_bits)
    walsh = WalshTransform(n_bits)
    N = domain.space_cardinality()
    base_truth_table = generate_alternate_balanced_binary_vector_one_zero(N)

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
    torus_dim = 0
    radius = 2
    pop_shape = (10, 10)
    pressure = 3
    cmp_rate = 1.0
    #sampling_probabilities = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0.0]
    sampling_probabilities = [0.7, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0]

    best_program, best_score = truth_tables_ea(
        n_bits=10,
        pop_size=50,
        n_iter=500,
        seed=1,
        verbose=True,
        pressure=pressure,
        torus_dim=torus_dim,
        radius=radius,
        pop_shape=pop_shape,
        cmp_rate=cmp_rate,
    )
    end_time = time()
    print(f"Best program: {best_program.genome}")
    print(f"Best score: {best_score}")
    print(f"Time taken: {(end_time - start_time) / 60} minutes")
    
    #best_program, best_score = programs_rs(n_bits=10, sampling_probabilities=sampling_probabilities, warm_up=50000, n_iter=500000, seed=1, min_length=2, max_length=40, verbose=True)
    #best_program, best_score = programs_sa(n_bits=10, sampling_probabilities=sampling_probabilities, warm_up=50000, n_iter=500000, seed=1, min_length=2, max_length=10, verbose=True)
    
    best_program, best_score = programs_ea(
        n_bits=10,
        sampling_probabilities=sampling_probabilities,
        pop_size=50,
        #warm_up=0,
        n_iter=500,
        seed=1,
        min_length=2,
        max_length=20,
        verbose=True,
        pressure=pressure,
        torus_dim=torus_dim,
        radius=radius,
        pop_shape=pop_shape,
        cmp_rate=cmp_rate
    )
    end_time = time()
    print(f"Best program: {best_program.genome}")
    print(f"Best score: {best_score}")
    print(f"Time taken: {(end_time - start_time) / 60} minutes")
