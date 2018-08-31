import pyquil.api as api
import numpy as np
from grove.pyqaoa.qaoa import QAOA
from pyquil.paulis import PauliTerm, PauliSum
import pyquil.quil as pq
from pyquil.gates import X
import itertools

import scipy.optimize
from . import utilities


class ForestTSPSolverNaive(object):
    def __init__(self, distance_matrix, steps=1, ftol=1.0e-2, xtol=1.0e-2, constraints=False):

        self.distance_matrix = distance_matrix
        self.number_of_nodes = len(self.distance_matrix)
        self.qvm = api.QVMConnection()
        self.steps = steps
        self.ftol = ftol
        self.xtol = xtol
        self.betas = None
        self.gammas = None
        self.qaoa_inst = None
        self.number_of_qubits = self.get_number_of_qubits()
        self.solution = None
        self.distribution = None
        self.most_frequent_string = None
        self.sampling_results = None
        self.constraints = constraints


        cost_operators = self.create_cost_operators()
        driver_operators = self.create_driver_operators()

        minimizer_kwargs = {'method': 'Nelder-Mead',
                                'options': {'ftol': self.ftol, 'xtol': self.xtol,
                                            'disp': False}}

        vqe_option = {'disp': print_fun, 'return_all': True,
                      'samples': None}

        qubits = list(range(self.number_of_qubits))

        self.qaoa_inst = QAOA(self.qvm, qubits, steps=self.steps, cost_ham=cost_operators,
                         ref_ham=driver_operators, store_basis=True,
                         minimizer=scipy.optimize.minimize,
                         minimizer_kwargs=minimizer_kwargs,
                         vqe_options=vqe_option)

    def solve_tsp(self):
        """
        Calculates the optimal angles (betas and gammas) for the QAOA algorithm 
        and returns a list containing the order of nodes.
        """
        self.find_angles()
        self.calculate_solution()
        return self.solution, self.distribution

    def find_angles(self):
        """
        Runs the QAOA algorithm for finding the optimal angles.
        """
        self.betas, self.gammas = self.qaoa_inst.get_angles()
        return self.betas, self.gammas

    def calculate_solution(self):
        """
        Samples the QVM for the results of the algorithm 
        and returns a list containing the order of nodes.
        """
        most_frequent_string, sampling_results = self.qaoa_inst.get_string(self.betas, self.gammas, samples=10000)
        self.most_frequent_string = most_frequent_string
        self.sampling_results = sampling_results
        self.solution = utilities.binary_state_to_points_order(most_frequent_string)
        
        all_solutions = sampling_results.keys()
        distribution = {}
        for sol in all_solutions:
            points_order_solution = utilities.binary_state_to_points_order(sol)
            distribution[tuple(points_order_solution)] = sampling_results[sol]
        self.distribution = distribution

    def create_cost_operators(self):
        cost_operators = []
        cost_operators += self.create_weights_cost_operators()
        if self.constraints:
            cost_operators += self.create_penalty_operators_for_bilocation()
            cost_operators += self.create_penalty_operators_for_repetition()

        return cost_operators

    def create_penalty_operators_for_bilocation(self):
        # Additional cost for visiting more than one node in given time t
        cost_operators = []
        number_of_nodes = len(self.distance_matrix)
        for t in range(number_of_nodes):
            range_of_qubits = list(range(t * number_of_nodes, (t + 1) * number_of_nodes))
            cost_operators += self.create_penalty_operators_for_qubit_range(range_of_qubits)
        return cost_operators

    def create_penalty_operators_for_repetition(self):
        # Additional cost for visiting given node more than one time
        cost_operators = []
        number_of_nodes = len(self.distance_matrix)
        for i in range(number_of_nodes):
            range_of_qubits = list(range(i, number_of_nodes**2, number_of_nodes))
            cost_operators += self.create_penalty_operators_for_qubit_range(range_of_qubits)
        return cost_operators

    def create_penalty_operators_for_qubit_range(self, range_of_qubits):
        cost_operators = []
        weight = -10 * np.max(self.distance_matrix)
        for i in range_of_qubits:
            if i == range_of_qubits[0]:
                z_term = PauliTerm("Z", i, weight)
                # all_ones_term = PauliTerm("I", 0, 0.5 * weight) - PauliTerm("Z", i, 0.5 * weight)
            else:
                z_term = z_term * PauliTerm("Z", i)
                # all_ones_term = all_ones_term * (PauliTerm("I", 0, 0.5) - PauliTerm("Z", i, 0.5))

        z_term = PauliSum([z_term])
        cost_operators.append(PauliTerm("I", 0, weight) - z_term)# - 2 * all_ones_term)

        return cost_operators

    def create_weights_cost_operators(self):
        cost_operators = []
        number_of_nodes = len(self.distance_matrix)
        
        for i in range(number_of_nodes):
            for j in range(i, number_of_nodes):
                for t in range(number_of_nodes - 1):
                    weight = -self.distance_matrix[i][j] / 2
                    if self.distance_matrix[i][j] != 0:
                        qubit_1 = t * number_of_nodes + i
                        qubit_2 = (t + 1) * number_of_nodes + j
                        cost_operators.append(PauliTerm("I", 0, weight) - PauliTerm("Z", qubit_1, weight) * PauliTerm("Z", qubit_2))

        return cost_operators

    def create_driver_operators(self):
        driver_operators = []
        
        for i in range(self.number_of_qubits):
            driver_operators.append(PauliSum([PauliTerm("X", i, -1.0)]))

        return driver_operators

    def get_number_of_qubits(self):
        return len(self.distance_matrix)**2


def print_fun(x):
    print(x)
        