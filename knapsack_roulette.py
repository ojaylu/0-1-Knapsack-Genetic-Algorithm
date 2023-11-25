import numpy as np
import time

P = 100 # population size
r = 2/3 # ratio of offspring
G = 100 # number of generations
m = 0.3 # mutation rate

def fitnessScaling(scores):
    f = lambda x: 1 if x == 0 else x
    vf = np.vectorize(f)
    scaled_scores = vf(scores)
    return scaled_scores/np.sum(scaled_scores)

def genericKnapsack(C,w,v,n):
    # initilization
    population = np.random.randint(2, size=(P, n))
    print(population)

    no_of_parents = int(P*r//2*2)
    elitism_index = no_of_parents

    evaluation_function = lambda y: np.apply_along_axis(lambda x: np.dot(x,v) if np.dot(x,w) <= C else 0, 1, y)

    for i in range(G):
        print(f"Generation {i}")
        # evaluation
        scores = evaluation_function(population)
        print("scores:", scores)
        print("max score:", scores.max(), ", quality:", scores.max()/40, ", mean:", np.mean(scores))

        new_population = np.zeros((P,n),dtype=int)
        indices = np.random.choice(P, size=no_of_parents, replace=False, p=fitnessScaling(scores))
        for j in range(no_of_parents//2):
            # selection
            parent1 = population[indices[2*j]]
            parent2 = population[indices[2*j+1]]

            # crossover
            crossover_index = np.random.randint(1,n)
            offspring1 = np.concatenate((parent1[:crossover_index], parent2[crossover_index:]))
            offspring2 = np.concatenate((parent2[:crossover_index], parent1[crossover_index:]))
            
            # mutation
            mutation1 = np.random.choice(2, size=n, p=[m,1-m])
            offspring1 = offspring1 * mutation1
            mutation2 = np.random.choice(2, size=n, p=[m,1-m])
            offspring2 = offspring2 * mutation2

            new_population[2*j] = offspring1
            new_population[2*j+1] = offspring2
        new_population[elitism_index:] = population[np.argsort(scores)][elitism_index:]
        population = new_population
    print(f"Generation {G}")
    final_scores = evaluation_function(population)
    print("scores:", final_scores)
    return [final_scores.max(), np.argmax(final_scores), population[np.argmax(final_scores)]]