from . import forest_tsp_solver_naive
from . import utilities
import os
from collections import Counter
import numpy as np
import pdb
import csv
import time

def analyze_singe_QAOA_run(distance_matrix, steps, tol, filename):
    results_file = open(filename, 'a')
    if os.stat(filename).st_size == 0:
        results_file.write("steps,tol,time,valid_prob,best_prob,best_valid,best_cost,optimal_cost\n")

    start_time = time.time()
    tsp_solver = forest_tsp_solver_naive.ForestTSPSolverNaive(distance_matrix, use_constraints=True)
    tsp_solver.solve_tsp()
    end_time = time.time()
    calculation_time = end_time - start_time
    sampling_results = tsp_solver.sampling_results
    distribution = sampling_results

    probability_of_valid = get_probability_of_valid_solutions(distribution)
    all_solutions_count = sum(distribution.values())
    best_solution = distribution.most_common()[0][0]
    best_valid = check_if_binary_solution_is_valid(best_solution)
    best_prob = distribution[best_solution] / all_solutions_count
    if best_valid:
        best_cost = calculate_cost_of_solution(best_solution, distance_matrix, is_binary=True)
    else:
        best_cost = np.nan

    optimal_solution = utilities.solve_tsp_brute_force(distance_matrix, starting_city=None, verbose=False)
    optimal_cost = calculate_cost_of_solution(optimal_solution, distance_matrix)
    params = [steps, tol]
    results = [calculation_time, probability_of_valid, best_prob, best_valid, best_cost, optimal_cost]
    csv_writer = csv.writer(results_file)
    csv_writer.writerow(params + results)
    results_file.close()

def get_probability_of_valid_solutions(distribution):
    valid_solutions_count = 0
    for solution in distribution:
        if check_if_binary_solution_is_valid(solution):
            valid_solutions_count += distribution[solution]
    all_solutions_count = sum(distribution.values())

    return valid_solutions_count / all_solutions_count

def check_if_binary_solution_is_valid(solution):
    number_of_nodes = int(np.sqrt(len(solution)))
    time_groups = [solution[number_of_nodes*i:number_of_nodes*(i+1)] for i in range(number_of_nodes)]
    for group in time_groups:
        if np.sum(group) != 1:
            return False
        if time_groups.count(group) != 1:
            return False
    return True

def calculate_cost_of_solution(solution, distance_matrix, is_binary=False):
    if is_binary:
        solution = utilities.binary_state_to_points_order(solution)
    return utilities.calculate_cost(distance_matrix, solution)

def main():
    cities = utilities.create_cities(3)
    distance_matrix = utilities.get_distance_matrix(cities)
    while True:
        steps = random.choice([1,2,3])
        tol = random.choice([10e-1, 10e-2, 10e-3, 10e-4])
        # for i in range(20):
        #     for steps in [1, 2, 3]:
        #         for tol in [10e-2, 10e-3, 10e-4]:
        filename = "results.csv"
        analyze_singe_QAOA_run(distance_matrix, steps, tol, filename)    

if __name__ == '__main__':
    main()