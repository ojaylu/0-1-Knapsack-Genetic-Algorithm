import numpy as np

P = 100 # population size
r = 2/3 # ratio of offspring
G = 50 # number of generations
m = 0.5 # mutation rate

def knapsack(C,w,v,n):
    # initilization
    population = np.random.randint(2, size=(P, n))
    print(population)

    reproduction_iteration = int(P*r//2)
    elitism_index = reproduction_iteration*2
    for i in range(G):
        print(f"Generation {i}")
        # evaluation
        scores = np.apply_along_axis(lambda x: np.dot(x,v) if np.dot(x,w) <= C else 0, 1, population)

        print(scores)
        new_population = np.zeros((P,n),dtype=int)

        for j in range(reproduction_iteration):
            # selection
            indices = np.random.choice(P, size=2, replace=False, p=scores/np.sum(scores))
            parent1 = population[indices[0]]
            parent2 = population[indices[1]]

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

if __name__ == '__main__': 
    profit = [10, 18, 12, 10, 15] 
    weight = [2, 3, 4, 5, 6] 
    C = 10
    n = len(profit)
    print(knapsack(C, weight, profit, n))