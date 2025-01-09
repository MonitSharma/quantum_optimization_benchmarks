"""

Maximum Independent Set

"""

# basic imports
import os
import json 
import io 
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
import pickle
from docplex.mp.model import Model





# quantum imports
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_algorithms.optimizers import ADAM, POWELL, SLSQP, COBYLA, P_BFGS
estimator = StatevectorEstimator()


# pce imports

sys.path.append(os.path.abspath(os.path.join('../..')))
from pce_qubo.pauli_correlation_encoding import PauliCorrelationEncoding
from pce_qubo.mixed_optimize import PauliCorrelationOptimizer

from pce_qubo.utility import Utility
from qiskit.quantum_info import SparsePauliOp, Statevector

import glob

# function to read the file

# the file path
directory_path = "../mis_small_datasets/"

# file_path = "../mis_datasets/1dc.64.txt"

file_paths = glob.glob(os.path.join(directory_path, "*.txt"))
for file_path in file_paths:
    print(f"Processing file: {file_path}")



    def read_graph_from_file(file_path):
        """
        Reads a graph from a text file and returns the number of nodes, edges, and edge list.

        Parameters:
        - file_path: str, path to the text file containing the graph definition.

        Returns:
        - num_nodes: int, number of nodes in the graph.
        - edges: list of tuple, list of edges in the graph.
        """
        edges = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if parts[0] == 'p':
                    num_nodes = int(parts[2])
                    num_edges = int(parts[3])
                elif parts[0] == 'e':
                    u = int(parts[1]) - 1  # Convert to 0-based index
                    v = int(parts[2]) - 1  # Convert to 0-based index
                    edges.append((u, v))
        return num_nodes, edges

    def create_max_independent_set_model(num_nodes, edges):
        """
        Creates a CPLEX model for the Maximum Independent Set (MIS) problem.

        Parameters:
        - num_nodes: int, the number of nodes in the graph.
        - edges: list of tuple, list of edges in the graph.

        Returns:
        - model: CPLEX model.
        - x: list of CPLEX binary variables representing node selection.
        """
        # Create CPLEX model
        model = Model(name="Maximum Independent Set")

        # Decision variables: x[i] = 1 if node i is in the independent set, 0 otherwise
        x = model.binary_var_list(num_nodes, name="x")

        # Objective: Maximize the number of selected nodes
        model.maximize(model.sum(x[i] for i in range(num_nodes)))

        # Constraints: At most one endpoint of each edge can be in the independent set
        for u, v in edges:
            model.add_constraint(x[u] + x[v] <= 1, f"edge_constraint_{u}_{v}")

        return model, x

    # Example usage
    if __name__ == "__main__":
        file_path = file_path  # Replace with the path to your graph file

        # Read the graph from the file
        num_nodes, edges = read_graph_from_file(file_path)

        # Create the MIS model
        model, x = create_max_independent_set_model(num_nodes, edges)

        # Print model summary
        # print(model.export_as_lp_string())

        # Solve the model
        # no need to solve as we already know the result
        # solution = model.solve()

        # if solution:
        #     print("Objective value (size of independent set):", solution.objective_value)
        #     independent_set = [i + 1 for i in range(num_nodes) if x[i].solution_value > 0.5]  # Convert back to 1-based index
        #     print("Nodes in the maximum independent set:", independent_set)
        # else:
        #     print("No solution found.")




    # make a quadratic program from the model
    qp = from_docplex_mp(model)

    # convert the quadratic program to a QUBO
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)

    # number of variables
    num_vars = qubo.get_num_vars()
    print('Number of variables:', num_vars)


    # call PCE
    pauli_encoder = PauliCorrelationEncoding()
    k = 2
    num_qubits = pauli_encoder.find_n(num_vars, k)
    pauli_strings = SparsePauliOp(pauli_encoder.generate_pauli_strings(num_qubits, num_vars, k))
    print(f"Number of qubits: {num_qubits} for k={k} with {num_vars} binary variables")


    depth =  2 * num_qubits
    ansatz = pauli_encoder.BrickWork(depth=depth, num_qubits=num_qubits)
    optimizer = COBYLA(maxiter=1000, tol=1e-6)

    pce = PauliCorrelationOptimizer(pauli_encoder=pauli_encoder, depth = depth,qubo=qubo, k=k,
                                    num_qubits=num_qubits,estimator=estimator)


    # run the optimizer
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Functions
    def adaptive_perturbation(params, perturbation_factor, failure_count, historical_trend, max_factor=1e-2):
        """Adds directional and random perturbation based on historical trends."""
        random_perturbation = np.random.normal(0, perturbation_factor * (1 + failure_count / 5), size=params.shape)
        directional_perturbation = perturbation_factor * np.sign(historical_trend)
        return params + random_perturbation + directional_perturbation

    def weighted_blended_initialization(previous_params, performance_weights, blend_factor=0.7):
        """Blends previous parameters with random ones, weighted by performance."""
        random_params = np.random.rand(len(previous_params))
        weighted_params = performance_weights * previous_params
        return blend_factor * weighted_params + (1 - blend_factor) * random_params

    def initialize_within_range(num_params, lower_bound=-np.pi, upper_bound=np.pi):
        """Initializes parameters uniformly within a specified range."""
        return np.random.uniform(lower_bound, upper_bound, size=num_params)

    def history_based_cooling_schedule(temperature, alpha, improvement_history, window=5):
        """Dynamically adjusts cooling based on recent improvement trends."""
        if len(improvement_history) >= window:
            recent_trend = sum(improvement_history[-window:])
            if recent_trend > 0:
                alpha = min(alpha * 1.05, 0.99)
            else:
                alpha = max(alpha * 0.95, 0.8)
        return max(temperature * alpha, 1e-8)

    def save_checkpoint(filename, params, cost, round_num):
        """Saves the current state of optimization."""
        checkpoint = {'params': params, 'cost': cost, 'round_num': round_num}
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(filename):
        """Loads a saved optimization state."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def evaluate_perturbed_params(params, perturbation_factor, optimizer, pce, no_improvement_count, historical_trend):
        """Evaluate a perturbed parameter set and return both parameters and QUBO cost."""
        perturbed_params = adaptive_perturbation(params, perturbation_factor, no_improvement_count, historical_trend)
        optimized_params = pce.optimize(optimizer, perturbed_params)  # Perform optimization

        # Evaluate the QUBO cost for the optimized parameters
        final_ansatz = pauli_encoder.BrickWork(depth=depth, num_qubits=num_qubits).assign_parameters(optimized_params)
        psi_final = Statevector(final_ansatz)
        utility = Utility()
        qubo_bitstring = utility.evaluate_sign_function(psi_final, pauli_strings)
        qubo_cost = qubo.objective.evaluate(qubo_bitstring)

        return optimized_params, qubo_cost

    # Initialization
    logger.info("Step 1: Initializing parameters and variables.")
    params = initialize_within_range(ansatz.num_parameters)
    best_params = params.copy()  # Store the best parameters
    best_qubo_cost = float('inf')  # Initialize best QUBO cost to infinity
    best_qubo_bitstring = None  # Initialize best QUBO bitstring
    no_improvement_count = 0  # Counter for consecutive rounds without improvement
    max_no_improvement_rounds = 10  # Stop after 10 consecutive rounds without improvement
    perturbation_factor = 1e-4  # Initial perturbation factor
    decay_factor = 0.99  # Decay factor for perturbation
    early_stopping_threshold = 1e-6  # Threshold for minimal improvement
    fixed_threshold = 50  # Fixed improvement threshold
    percentage_threshold = 0.001  # 0.1% of the best QUBO cost
    improvement_history = []  # Track improvements for plotting
    cumulative_improvement = 0  # Track cumulative improvement
    historical_trend = np.zeros_like(params)  # Track parameter improvement trend
    performance_weights = np.ones_like(params)  # Initialize performance weights

    # Logging optimizer information
    # logger.info(f"Using {optimizer.__class__.__name__} optimizer.")
    # logger.info(f"Early stopping configured with fixed_threshold={fixed_threshold} and percentage_threshold={percentage_threshold * 100:.2f}%.")

    round_num = 0
    last_improvement_round = 0  # Track the round where the last significant improvement occurred

    while no_improvement_count < max_no_improvement_rounds:
        round_num += 1
        logger.info(f"\n--- Round {round_num} ---")

        # Evaluate perturbations sequentially
        best_parallel_cost = float('inf')
        best_result = None

        for _ in range(4):  # Evaluate 4 perturbations sequentially
            result, qubo_cost = evaluate_perturbed_params(
                params, perturbation_factor, optimizer, pce, no_improvement_count, historical_trend
            )
            if qubo_cost < best_parallel_cost:
                best_parallel_cost = qubo_cost
                best_result = result

        result = best_result  # Use the best result from evaluations

        # Prepare the ansatz with the optimized parameters
        # logger.info("Assigning optimized parameters to the ansatz...")
        final_ansatz = pauli_encoder.BrickWork(depth=depth, num_qubits=num_qubits).assign_parameters(result)
        psi_final = Statevector(final_ansatz)

        # Evaluate the QUBO cost and bitstring
        # logger.info("Evaluating the QUBO cost and bitstring...")
        utility = Utility()
        qubo_bitstring = utility.evaluate_sign_function(psi_final, pauli_strings)
        qubo_cost = qubo.objective.evaluate(qubo_bitstring)
        # logger.info(f"QUBO cost for this round: {qubo_cost}")
        market_share_bitstring = converter.interpret(qubo_bitstring)
        initial_feasible = qp.get_feasibility_info(market_share_bitstring)[0]
        market_share_cost = qp.objective.evaluate(market_share_bitstring)
        logger.info(f"MIS cost: {market_share_cost}")
        logger.info(f"Initial feasibility: {initial_feasible}")

        # Check if the QUBO cost improved
        improvement = best_qubo_cost - qubo_cost
        improvement_history.append(improvement)
        cumulative_improvement += improvement
        historical_trend = np.sign(result - params) * improvement

        # Calculate dynamic threshold
        dynamic_threshold = max(fixed_threshold, percentage_threshold * best_qubo_cost)

        # Check if |qubo_cost| equals market_share_cost
        if abs(qubo_cost) == market_share_cost:
            logger.info(f"Stopping condition met: |QUBO cost| equals Market Share Cost.")
            logger.info(f"Final QUBO cost: {qubo_cost}, MIS Cost: {market_share_cost}.")
            break

        if qubo_cost < best_qubo_cost:
            # logger.info(f"Improvement detected! Previous best QUBO cost: {best_qubo_cost}, Current QUBO cost: {qubo_cost}")
            best_qubo_cost = qubo_cost
            best_params = result.copy()  # Save the best parameters
            best_qubo_bitstring = qubo_bitstring.copy()  # Save the best bitstring
            no_improvement_count = 0  # Reset no improvement counter
            perturbation_factor *= decay_factor  # Decay perturbation factor for precision
            last_improvement_round = round_num  # Update last improvement round

            # Perturb the trained parameters slightly
            # logger.info("Perturbing the best parameters slightly for the next round.")
            params = adaptive_perturbation(best_params, perturbation_factor, no_improvement_count, historical_trend)
        else:
            # logger.info(f"No improvement this round. Previous best QUBO cost remains: {best_qubo_cost}")
            no_improvement_count += 1

            # Apply perturbation or blended initialization
            if no_improvement_count % 2 == 0:
                # logger.info("Applying blended initialization to explore new solutions.")
                params = weighted_blended_initialization(best_params, performance_weights)
            else:
                # logger.info("Applying stronger perturbation to the best parameters.")
                params = adaptive_perturbation(best_params, perturbation_factor * 2, no_improvement_count, historical_trend)

        # Stop if no improvement for max_no_improvement_rounds consecutive rounds
        rounds_remaining = max_no_improvement_rounds - no_improvement_count
        logger.info(f"Consecutive no-improvement rounds: {no_improvement_count}. Rounds remaining before stopping: {rounds_remaining}.")

        if no_improvement_count >= max_no_improvement_rounds:
            logger.info(f"No improvement detected for {max_no_improvement_rounds} consecutive rounds.")
            logger.info(f"Early stopping triggered at round {round_num}.")
            break

        # Save checkpoints
        save_checkpoint("optimization_checkpoint.pkl", best_params, best_qubo_cost, round_num)

    # Final Results
    logger.info("\nOptimization complete.")
    logger.info(f"Best QUBO cost: {best_qubo_cost}")
    logger.info(f"Best QUBO bitstring: {best_qubo_bitstring}")
    market_share_bitstring = converter.interpret(best_qubo_bitstring)
    initial_feasible = qp.get_feasibility_info(market_share_bitstring)[0]
    market_share_cost = qp.objective.evaluate(market_share_bitstring)
    logger.info(f"Market share cost: {market_share_cost}")


    logger.info(f"Final feasibility: {initial_feasible}")



    def combined_bit_search(qubo_bitstring, qubo):
        """
        Performs a combined bit-flip and bit-swap search on a given QUBO bitstring.

        Parameters:
        - qubo_bitstring (list): A binary list representing the current solution.
        - qubo: An object with a callable `objective.evaluate()` function to evaluate solutions.

        Returns:
        - best_bitstring (list): The best bitstring found during the search.
        - best_cost (float): The cost associated with the best bitstring.
        """
        import copy

        def evaluate_cost(bitstring):
            """Helper to evaluate the cost of a bitstring."""
            return qubo.objective.evaluate(bitstring)

        # Initialize with the current solution
        best_bitstring = qubo_bitstring[:]
        best_cost = evaluate_cost(qubo_bitstring)
        n = len(qubo_bitstring)

        print("Starting cost:", best_cost)

        # Perform single-bit flips
        for i in range(n):
            # Flip the i-th bit
            flipped_bitstring = best_bitstring[:]
            flipped_bitstring[i] = 1 - flipped_bitstring[i]

            # Evaluate the new cost
            new_cost = evaluate_cost(flipped_bitstring)

            # Update the best solution if the new one is better
            if new_cost < best_cost:
                best_bitstring = flipped_bitstring
                best_cost = new_cost
                # print(f"Bit flip: Improved solution found by flipping bit {i}: Cost = {best_cost}")

        # Perform two-bit swaps
        for i in range(n):
            for j in range(i + 1, n):  # Avoid redundant swaps
                # Swap the i-th and j-th bits
                swapped_bitstring = best_bitstring[:]
                swapped_bitstring[i] = 1 - swapped_bitstring[i]
                swapped_bitstring[j] = 1 - swapped_bitstring[j]

                # Evaluate the new cost
                new_cost = evaluate_cost(swapped_bitstring)

                # Update the best solution if the new one is better
                if new_cost < best_cost:
                    best_bitstring = swapped_bitstring
                    best_cost = new_cost
                    # print(f"Bit swap: Improved solution found by swapping bits {i} and {j}: Cost = {best_cost}")

        # print("Final best cost:", best_cost)
        return best_bitstring, best_cost


    # Example input bitstring (current solution)
    initial_bitstring = best_qubo_bitstring
    # print("Initial bitstring:", initial_bitstring)

    # Initialize the QUBO object
    qubo = qubo

    # Perform the combined search
    best_solution, best_cost = combined_bit_search(initial_bitstring, qubo)

    print("\nBest Solution:", best_solution)
    print("Best Cost:", best_cost)

    market_share_bitstring = converter.interpret(best_solution)
    initial_feasible = qp.get_feasibility_info(market_share_bitstring)[0]
    market_share_cost = qp.objective.evaluate(market_share_bitstring)
    print(f"MDKP cost: {market_share_cost} and Feasibility: {initial_feasible}")

    print("----------------------------------------------------")
    print("----------------------------------------------------")
    # print new line for better readability
    print("\n")



