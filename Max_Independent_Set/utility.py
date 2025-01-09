import numpy as np
import random
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
class MaxCutUtility:
    def __init__(self, weight_matrix):
        self.weight_matrix = weight_matrix

    @staticmethod
    def evaluate_sign_function(psi, operators):
        """
        Evaluate the sign function and binary values for each operator.

        Args:
            psi (Statevector): Quantum state vector.
            operators (list of SparsePauliOp): List of operators.

        Returns:
            list of int: List of binary values (0 or 1) for each operator.
        """
        binary_values = []

        for op in operators:
            pi_i = psi.expectation_value(op).real
            x_i = np.sign(pi_i)
            if x_i < 0:
                x_i = 0
            else:
                x_i = 1
            binary_values.append(x_i)

        return binary_values

    @staticmethod
    def evaluate_max_cut(solution, weight_matrix):
        """
        Evaluate the max cut value for a given solution.

        Args:
            solution (list of int): Binary list representing the solution.
            weight_matrix (numpy.ndarray): Weight matrix representing the graph.

        Returns:
            float: The value of the max cut.
        """
        cut_value = 0
        n = len(solution)
        for i in range(n):
            for j in range(i + 1, n):
                if solution[i] != solution[j]:
                    cut_value += weight_matrix[i, j]
        return cut_value
    

    @staticmethod
    def max_cut_to_qubo_solution(cut_solution):
        """
        Convert the solution of the weighted max cut to the original QUBO solution.

        Parameters:
        cut_solution (list): A binary list representing the cut solution.

        Returns:
        list: The solution to the original QUBO problem.
        """
        n = len(cut_solution) - 1  # Subtract 1 because cut_solution includes the extra 0th node
        qubo_solution = []

        for i in range(1, n + 1):
            if cut_solution[i] != cut_solution[0]:
                qubo_solution.append(0)
            else:
                qubo_solution.append(1)

        return qubo_solution

    def evaluate_initial_score(self, psi, operators):
        """
        Evaluate the initial score for the max-cut solution.

        Args:
            psi (Statevector): Quantum state vector.
            operators (list of SparsePauliOp): List of operators.

        Returns:
            float: Initial max-cut score.
        """
        max_cut_solution_pce = self.evaluate_sign_function(psi, operators)
        initial_score = self.evaluate_max_cut(max_cut_solution_pce, self.weight_matrix)
        return initial_score, max_cut_solution_pce


