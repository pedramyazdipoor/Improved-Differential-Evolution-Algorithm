#!pip install benchmark-functions
import numpy as np
import benchmark_functions as bf
import random
from random import randint
import matplotlib.pyplot as plt
from numpy import save
from numpy import load
import math
import time
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
#====================================================================================================================================
np.seterr(divide='ignore')

def f1(list):
    result = 0
    for i in range(0, len(list)):
        result += list[i] ** 2
    return result


def f2(list):
    result1, result2 = 0, 1
    for i in range(0, len(list)):
        gene = np.abs(list[i])
        result1 += gene
        result2 *= gene
    return result1 + result2


def f3(list):
    result = 0
    for i in range(0, len(list)):
        result += np.sum(list[0:i + 1]) ** 2
    return result


def f4(list):
    result = 0
    for i in range(0, len(list)):
        result += (i + 1) * (list[i] ** 2)
    return result


def f8(chromosome):
    d = len(chromosome)
    partA = 0
    partB = 1
    count = 0
    for i in chromosome:
        partA += (i**2)
        partB *= math.cos(float(i) / math.sqrt(count+1))
        count += 1
    return 1 + (float(partA)/4000.0) - float(partB)


def f9(list):
    result = 0
    for i in range(0, len(list)):
        result += (list[i] + 0.5) ** 2
    return result


def f10(list):
    result = 0
    for i in range(0, len(list)):
        result += (i + 1) * (list[i] ** 4) + random.random()
    return result


def f11(list):
    result = 0
    for i in range(0, len(list)):
        x = list[i]
        result += np.abs((x * np.sin(x)) + (0.1 * x))
    return result


def f12(list):
    result = 0
    for i in range(0, len(list)):
        result += (np.abs(list[i])) ** (i + 2)
    return result

#####################=================================== PARAMETERS ==================================================================
cr = 0.1                  #crossover rate
F1, F2 = 0.3 , 0.7        #scaling factors
jr = 0.5                  #jumping rate
#dimensions = [10,30,50,100,200,500,1000]
dimension = 30
pop_size = 100
generations = 1500
function_number = 8     #index == function number (12 functions, 1-12)
#=================== DONOT TOUCH HERE PLEASE =========================================================================================
func7 = bf.Ackley(n_dimensions=dimension)     #f7
func5 = bf.Rosenbrock(n_dimensions=dimension) #f5
func6 = bf.Rastrigin(n_dimensions=dimension)  #f6
func8 = bf.Griewank(n_dimensions=dimension)   #f8
func_list = [0,f1,f2,f3,f4,func5,func6,func7,func8,f9,f10,f11,f12]
func = func_list[function_number]
absolute_bound = [0,100,10,100,1.28,30,5.12,32,600,100,1.28,10,1] #index == function_number
lower_bound = -1*(absolute_bound[function_number])
upper_bound = (absolute_bound[function_number])
#=====================================================================================================================================
#first population
def initialization(dimension, pop_size, lower_bound, upper_bound):
    pop = []
    for i in range(0, pop_size):
        temp = []
        for j in range(0, dimension):
            temp.append(random.uniform(lower_bound, upper_bound))
        pop.append(temp)
    return pop
#====================================================================================================================================
def fitness_evaluator(pop):
    dimension = len(pop[0])
    fitness_list = []
    for i in range(0, len(pop)):
        #fitness_list.append([f3(pop[i]), i])
        fitness_list.append([func(pop[i]),i])
    return fitness_list
#====================================================================================================================================
#select best individuals randomly
def selector(pop, lowRank):
    start = lowRank - 1
    super_pop = []
    end = randint(10, 30)
    for i in range(start, start + end):
        super_pop.append(pop[i])
    return super_pop
#====================================================================================================================================
def opposition_learning(pop):
    dim = len(pop[0])
    a, b = [0 for j in range(0, dim)], [0 for k in range(0, dim)]
    for i in range(0, dim):
        min = pop[0][i]
        max = pop[0][i]
        a[i] = min
        b[i] = max
        for j in range(1, len(pop)):
            temp = pop[j][i]
            if temp < min:
                a[i] = temp
            elif temp > max:
                b[i] = temp
    for k in range(0, len(pop)):
        chr = [0 for t in range(0, dim)]
        for h in range(0, dim):
            chr[h] = a[h] + b[h] - pop[k][h]
        pop.append(chr)
    list = fitness_evaluator(pop)
    list.sort()
    result = []
    for z in range(0, int(len(pop) / 2)):
        result.append(pop[list[z][-1]])
    return result
#====================================================================================================================================
def crossover(pop, x1, x2, x3, best):
    dim = len(pop[0])
    new_pop = []
    for i in range(0, len(pop)):
        x = pop[i]
        x_new = []
        r = randint(0, dim - 1)
        for j in range(0, dim):
            rand = random.random()
            if (rand < cr) or j == r:
                x_new.append(x3[j] + F1 * (best[j] - x3[j]) + F2 * (x1[j] - x2[j])) #mutation vector
            else:
                x_new.append(x[j])
        new_pop.append(x_new)
    fit_pop = fitness_evaluator(pop)
    fit_new_pop = fitness_evaluator(new_pop)
    result = []
    for u in range(0, len(pop)):
        if fit_pop[u] < fit_new_pop[u]:
            result.append(pop[u])
        else:
            result.append(new_pop[u])
    return result
#====================================================================================================================================
#main function
def execute(rank, n_executions):
  #list_gen = [0 for u in range(0,1500)]
  list_n = [0 for t in range(0,n_executions)]
  list_time = [0 for t in range(0,n_executions)]
  for n in range(0,n_executions):
    start = time.time()
    pop = initialization(dimension,pop_size,lower_bound,upper_bound)
    pop = opposition_learning(pop)
    list = fitness_evaluator(pop)
    list.sort()
    for g in range(0,generations):
    #================ best member ===================================
      x_best = list[0]
      del list[0]
    #================ x3 member =====================================
      super = selector(list,rank)
      x3_index = randint(0, len(super)-1)
      x3 = super[x3_index]
      for s in range(0,len(list)):
        if list[s][-1] == x3[-1] :
          del list[s]
          break
    #================ x1 member =====================================
      x1_index = randint(0, len(list)-1)
      x1 = list[x1_index]
      del list[x1_index]
    #================ x2 member =====================================
      x2_index = randint(0, len(list)-1)
      x2 = list[x2_index]
      del list[x2_index]
      pop = crossover(pop,pop[x1[-1]],pop[x2[-1]],pop[x3[-1]],pop[x_best[-1]])
      if random.random() < jr : pop = opposition_learning(pop)
      list = fitness_evaluator(pop)
      list.sort()
      #print(list[0][0])            #best fitness in every generation
      
    list_n[n] = list[0][0]          #best fitness in a single run
    end = time.time()
    list_time[n] = end - start      #execution time in a single run
    
  print("best : ",min(list_n))
  print("worst: ",max(list_n))
  print("mean: ",np.average(list_n))
  print("median:",np.median(list_n))
  print("avg_time:",np.average(list_time))
  print("std_time:",np.std(list_time))
#====================================================================================================================================
#Here you can easily run with desired parameters 
#rank = [1,11,21,31]
execute(rank = 1, n_executions = 10) 
#====================================================================================================================================
#code of drawing figures and graphs
'''
one = load('list_gen_1.npy')
two = load('list_gen_11.npy')
three = load('list_gen_21.npy')
four = load('list_gen_31.npy')
time = np.arange(1,1501)
plt.plot(time, one,label='1_30')
plt.plot(time, two,label='11_40')
plt.plot(time, three,label='21_50')
plt.plot(time, four,label='31_60')
plt.legend()
plt.xlabel('Generations')
plt.ylabel('Avg fitness (log)')
plt.title("f7")
plt.savefig("f7.png")'''
