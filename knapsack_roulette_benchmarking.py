import numpy as np
import time
import pandas as pd

P = 100 # population size
r = 2/3 # ratio of offspring
G = 100 # number of generations
m = 0.3 # mutation rate

def fitnessScaling(scores):
    f = lambda x: 1 if x == 0 else x
    vf = np.vectorize(f)
    scaled_scores = vf(scores)
    return scaled_scores/np.sum(scaled_scores)

def geneticKnapsack(C,w,v,n):
    # initilization
    population = np.random.randint(2, size=(P, n))

    no_of_parents = int(P*r//2*2)
    elitism_index = no_of_parents

    evaluation_function = lambda y: np.apply_along_axis(lambda x: np.dot(x,v) if np.dot(x,w) <= C else 0, 1, y)

    for _ in range(G):
        # evaluation
        scores = evaluation_function(population)

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
    final_scores = evaluation_function(population)
    return final_scores.max()

def recursionKnapsack(C, w, v, n): 
    # Base Case
    if n == 0 or C == 0: 
        return 0
    
    if (w[n-1] > C): 
        return recursionKnapsack(C, w, v, n-1) 
  
    else: 
        return max( 
            v[n-1] + recursionKnapsack( 
                C-w[n-1], w, v, n-1), 
            recursionKnapsack(C, w, v, n-1)) 

if __name__ == '__main__':
    initital_size = 10
    final_size = 100
    step = 2

    number_of_records = (final_size - initital_size)//step + 1
    genetic_time = np.empty(number_of_records)
    recursion_time = np.empty(number_of_records)
    quality = np.empty(number_of_records)
    sizes = np.arange(initital_size, final_size+1, step)
    for i in range(number_of_records):
        genetic_trials = np.empty(5)
        recursion_trials = np.empty(5)
        quality_trials = np.empty(5)
        size = initital_size + i * step
        print(f"running {size}")
        for j in range(5):
            profit = np.random.randint(1, 100, size=size)
            weight = np.random.randint(1, size, size=size)
            C = size
            n = size
            print("\trunning genetic")
            start1 = time.time()
            generic = geneticKnapsack(C, weight, profit, n)
            end1 = time.time()
            elapsed1 = end1 - start1
            genetic_trials[j] = elapsed1

            print("\trunning recursion")
            start2 = time.time()
            recursion = recursionKnapsack(C, weight, profit, n)
            end2 = time.time()
            elapsed2 = end2 - start2
            recursion_trials[j] = elapsed2

            quality_trials[j] = generic/recursion
        genetic_time[i] = np.median(genetic_trials)
        recursion_time[i] = np.median(recursion_trials)
        quality[i] = np.median(quality_trials)

    print("writing to excel")
    data = {'genetic_time': genetic_time, 'recursion_time': recursion_time, 'quality': quality, "size": sizes}
    df = pd.DataFrame(data)
    df.to_excel("benchmarking.xlsx", index=False)
    print("done")