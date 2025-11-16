import numpy as np
import time
import random
from simpleai.search import SearchProblem, simulated_annealing



class KnapsackProblem(SearchProblem):
   def __init__(self, weights, values, capacity):
       self.weights = weights
       self.values = values
       self.capacity = capacity
       initial_state = tuple([0 for _ in range(len(weights))])
       super().__init__(initial_state)


   def actions(self, state):
       return list(range(len(state)))


   def result(self, state, action):
       """Kết quả: trạng thái mới sau khi lật bit tại vị trí 'action'"""
       new_state = list(state)
       new_state[action] = 1 - new_state[action]
       return tuple(new_state)


   def value(self, state):
       """Hàm đánh giá (fitness) cho trạng thái: giá trị tổng cộng"""
       total_weight = sum(w * s for w, s in zip(self.weights, state))
       total_value = sum(v * s for v, s in zip(self.values, state))
      

       if total_weight > self.capacity:
           total_value -= (total_weight - self.capacity) * 10
      
       return total_value





def run_SA(weights, values, capacity, iterations_limit=5000):
   problem = KnapsackProblem(weights, values, capacity)
   start_time = time.time()
   result = simulated_annealing(problem, iterations_limit=iterations_limit)
   elapsed = time.time() - start_time


   best_state = result.state
   total_weight = sum(w * s for w, s in zip(weights, best_state))
   total_value = sum(v * s for v, s in zip(values, best_state))
  
   return best_state, total_value, total_weight, elapsed, iterations_limit


def run_BCO(weights, values, capacity, num_bees=30, num_iterations=200):
   start_time = time.time()
   n = len(weights)
   population = np.random.randint(0, 2, (num_bees, n))


   def fitness(state):
       w = np.sum(np.array(weights) * state)
       v = np.sum(np.array(values) * state)
       if w > capacity:
           v -= (w - capacity) * 10
       return v


   best_solution = population[0].copy()
   best_fitness = fitness(best_solution)


   for _ in range(num_iterations):
       fitness_values = np.array([fitness(s) for s in population])
      
       # Chọn lựa tổ ong (solution) dựa trên độ thích nghi
       probs = (fitness_values - fitness_values.min() + 1e-6)
       probs = probs / probs.sum()


       new_population = []
       for _ in range(num_bees):
           j = np.random.choice(range(num_bees), p=probs)
           candidate = population[j].copy()
           flip = np.random.randint(0, n)
           candidate[flip] = 1 - candidate[flip]
           new_population.append(candidate)
       population = np.array(new_population)


       for s in population:
           f = fitness(s)
           if f > best_fitness:
               best_fitness = f
               best_solution = s.copy()


   elapsed = time.time() - start_time
   total_weight = np.sum(np.array(weights) * best_solution)
  
   complexity = num_bees * num_iterations
   return tuple(best_solution), best_fitness, total_weight, elapsed, complexity


def run_GA(weights, values, capacity, pop_size=30, generations=100, mutation_rate=0.1):
   start_time = time.time()
   n = len(weights)
   population = np.random.randint(0, 2, (pop_size, n))


   def fitness(state):
       w = np.sum(np.array(weights) * state)
       v = np.sum(np.array(values) * state)
       if w > capacity:
           v -= (w - capacity) * 10
       return v


   best_solution = population[0].copy()
   best_fitness = fitness(best_solution)


   for _ in range(generations):
       fitness_values = np.array([fitness(s) for s in population])
      
       parents_idx = np.argsort(fitness_values)[-pop_size // 2:]
       parents = population[parents_idx]


       children = []
       for _ in range(pop_size // 2):
           p1, p2 = parents[np.random.randint(0, len(parents), 2)]
           point = np.random.randint(1, n - 1)
           child = np.concatenate([p1[:point], p2[point:]])
          
           if np.random.rand() < mutation_rate:
               flip = np.random.randint(0, n)
               child[flip] = 1 - child[flip]
           children.append(child)


       population = np.vstack((parents, children))
      
       for s in population:
           f = fitness(s)
           if f > best_fitness:
               best_fitness = f
               best_solution = s.copy()


   elapsed = time.time() - start_time
   total_weight = np.sum(np.array(weights) * best_solution)
  
   complexity = generations * pop_size
   return tuple(best_solution), best_fitness, total_weight, elapsed, complexity



def random_dataset():
   """Tạo dữ liệu ngẫu nhiên cho bài toán Knapsack"""
   n = random.randint(5, 12)
   weights = [random.randint(5, 40) for _ in range(n)]
   values = [random.randint(10, 100) for _ in range(n)]
   capacity = random.randint(sum(weights)//3, sum(weights)//2)
   return weights, values, capacity


