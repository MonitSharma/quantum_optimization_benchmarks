# necessary imports
import os
import json
import glob
import sys
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt 
from docplex.mp.model import Model

# quantum imports
import qiskit
print("Qiskit Version:",qiskit.__version__)
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.compiler import transpile
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.applications import Knapsack, Maxcut
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA,POWELL,SLSQP,P_BFGS,ADAM,SPSA
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimator, BackendSampler
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2

backend = AerSimulator(method='matrix_product_state')
# qrao imports
from qiskit_optimization.algorithms.qrao import QuantumRandomAccessEncoding
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit_optimization.algorithms.qrao import (
    QuantumRandomAccessOptimizer,
    SemideterministicRounding,
)
from qiskit_optimization.algorithms.qrao import MagicRounding
output_folder = f"graph_shrinking"  # Folder name based on num_products and seed
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist


estimator = BackendEstimatorV2(backend=backend)
sampler = BackendSamplerV2(backend=backend)

# shrinking imports
from qubo_to_maxcut import *
from utility import *
from shrinking import *



"""
Helper functions required for parsing data files
"""




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




"""
Helper Quantum Functions
"""

# function to compute the CVaR

def compute_cvar(probabilities, values, alpha):
    """
    Computes the Conditional Value at Risk (CVaR) for given probabilities, values, and confidence level.
    CVaR is a risk assessment measure that quantifies the expected losses exceeding the Value at Risk (VaR) at a given confidence level.
    Args:
    probabilities (list or array): List or array of probabilities associated with each value.
    values (list or array): List or array of corresponding values.
    alpha (float): Confidence level (between 0 and 1).
    float: The computed CVaR value.
    Example:
    >>> probabilities = [0.1, 0.2, 0.3, 0.4]
    >>> values = [10, 20, 30, 40]
    >>> alpha = 0.95
    >>> compute_cvar(probabilities, values, alpha)
    35.0
    Notes:
    - The function first sorts the values and their corresponding probabilities.
    - It then accumulates the probabilities until the total probability reaches the confidence level alpha.
    - The CVaR is calculated as the weighted average of the values, considering only the top (1-alpha) portion of the distribution.
    
    Auxilliary method to computes CVaR for given probabilities, values, and confidence level.
    
    Attributes:
    - probabilities: list/array of probabilities
    - values: list/array of corresponding values
    - alpha: confidence level
    
    Returns:
    - CVaR
    """
    sorted_indices = np.argsort(values)
    probs = np.array(probabilities)[sorted_indices]
    vals = np.array(values)[sorted_indices]
    cvar = 0
    total_prob = 0
    for i, (p, v) in enumerate(zip(probs, vals)):
        done = False
        if p >= alpha - total_prob:
            p = alpha - total_prob
            done = True
        total_prob += p
        cvar += p * v
    cvar /= total_prob
    return cvar


# function to evaluate the bitstring

def eval_bitstring(H, x):
    """
    Evaluate the objective function for a given bitstring.
    
    Args:
        H (SparsePauliOp): Cost Hamiltonian.
        x (str): Bitstring (e.g., '101').
    
    Returns:
        float: Evaluated objective value.
    """
    # Translate the bitstring to spin representation (+1, -1)
    spins = np.array([(-1) ** int(b) for b in x[::-1]])
    value = 0.0

    # Loop over Pauli terms and compute the objective value
    for pauli, coeff in zip(H.paulis, H.coeffs):
        weight = coeff.real  # Get the real part of the coefficient
        z_indices = np.where(pauli.z)[0]  # Indices of Z operators in the Pauli term
        contribution = weight * np.prod(spins[z_indices])  # Compute contribution
        value += contribution

    return value









directory_path = "../mis_small_datasets/"
# Get all .dat files in the directory
file_paths = glob.glob(os.path.join(directory_path, "*.txt"))
for file_path in file_paths:
    print(f"Processing {file_path}")

    num_nodes, edges = read_graph_from_file(file_path)
    model, x = create_max_independent_set_model(num_nodes, edges)

    solution = model.solve()

    if solution:
        print("Objective value (size of independent set):", solution.objective_value)
        independent_set = [i + 1 for i in range(num_nodes) if x[i].solution_value > 0.5]  # Convert back to 1-based index
        print("Nodes in the maximum independent set:", independent_set)
    else:
        print("No solution found.")

    optimal_value = solution.objective_value if solution else None
    print(f"Optimal value: {optimal_value}")

    qp = from_docplex_mp(model)                 # made a quadratic program (qp)
    converter = QuadraticProgramToQubo()        # converter for qp to qubo  
    qubo = converter.convert(qp)                # the qubo

    num_vars = qubo.get_num_vars()
    print("Number of Variables in QUBO:",num_vars)

    """
    QUBO to Maxcut
    """
    linear = qubo.objective.linear.to_array()
    quadratic = qubo.objective.quadratic.to_array()
    weight_max_cut_qubo = QUBO(quadratic, linear)
    weight_max_cut_qubo.linear_to_square()
    max_cut_graph = weight_max_cut_qubo.to_maxcut()
    max_cut = Maxcut(max_cut_graph)
    problem_max_cut = max_cut.to_quadratic_program()


    graph = create_graph_from_weight_matrix(max_cut_graph)
    
    # number of nodes
    n = len(max_cut_graph)
    print("Number of Nodes: ",n)

    """ 
    SDP Correlations Calculation
    """

    # solve it via SDP Correlation

    B, X_opt = max_cut_sdp(graph)  # Solve SDP relaxation
    initial_correlations = calculate_sdp_correlations(B)  # Compute initial correlations

    # Shrink the graph to 3 nodes
    target_nodes = round(2/3 * num_vars)
    print("Graph Shrinking to ", target_nodes)
    # Shrink the graph
    shrunk_graph, shrunk_partition, shrinking_steps = shrink_graph_with_tracking(
        graph=graph,
        correlations=initial_correlations,
        target_num_nodes=target_nodes,
        recalculation_steps=1
    )

    # Visualize the resulting shrunk graph
    # plot_graph_step(shrunk_graph, f"Graph Shrunk to {target_nodes} Nodes", shrunk_partition)

    # Assuming `G` is the original graph and `shrunk_graph` is the reduced graph
    node_map = generate_node_map(graph, shrunk_graph)

    weight_matrix, node_map = get_weight_matrix(shrunk_graph)


    # create a maxcut problem from a graph
    maxcut = Maxcut(weight_matrix)
    qp_shrunk = maxcut.to_quadratic_program()


    converter_shrunk = QuadraticProgramToQubo()
    qubo_shrunk = converter_shrunk.convert(qp_shrunk)
    

    num_qubits = qubo_shrunk.get_num_vars()
    print(f"Number of qubits required: {num_qubits}")

    H, offset = qubo_shrunk.to_ising()
    
    alphas =[0.10] # confidence levels to be evaluated

    class Objective:
        """
        Wrapper for objective function to track the history of evaluations.
        """
        def __init__(self, H, offset, alpha, num_qubits, optimal=None):
            self.history = []
            self.H = H
            self.offset = offset
            self.alpha = alpha
            self.num_qubits = num_qubits
            self.optimal = optimal
            self.opt_history = []
            self.last_counts = {}  # Store the counts from the last circuit execution
            self.counts_history = []  # New attribute to store counts history

        def evaluate(self, thetas):
            """
            Evaluate the CVaR for a given set of parameters.
            """
            
            """ 
            Other Attempt 
            """
            # Create a new circuit
            qc = QuantumCircuit(num_qubits)

            # Create a single ParameterVector for all parameters
            theta = ParameterVector('theta', 2 * reps * num_qubits)

            # Build the circuit
            for r in range(reps):
                # Rotation layer of RY gates
                for i in range(num_qubits):
                    qc.ry(theta[r * 2 * num_qubits + i], i)
                
                # Rotation layer of RZ gates
                for i in range(num_qubits):
                    qc.rz(theta[r * 2 * num_qubits + num_qubits + i], i)
                
                # Entangling layer of CNOT gates
                if r < reps - 1:  # Add entanglement only between layers
                    for i in range(num_qubits - 1):
                        qc.cx(i, i + 1)

            # Add measurement gates
            qc.measure_all()
            qc = qc.assign_parameters(thetas)


            
            ############


            job = sampler.run([qc])
            result = job.result()
            data_pub = result[0].data
            self.last_counts = data_pub.meas.get_counts()  # Store counts
            self.counts_history.append(self.last_counts)
            
            # Evaluate counts
            probabilities = np.array(list(self.last_counts.values()), dtype=float)  # Ensure float array
            values = np.zeros(len(self.last_counts), dtype=float)
            
            for i, x in enumerate(self.last_counts.keys()):
                values[i] = eval_bitstring(self.H, x) + self.offset
            
            # Normalize probabilities
            probabilities /= probabilities.sum()  # No more dtype issue

            # Track optimal probability
            if self.optimal:
                indices = np.where(values <= self.optimal + 1e-8)
                self.opt_history.append(sum(probabilities[indices]))
            
            # Compute CVaR
            cvar = compute_cvar(probabilities, values, self.alpha)
            self.history.append(cvar)
            return cvar
    reps = 5    
    num_params = 2 * reps * num_qubits
    initial_params = np.random.rand(num_params) * 2 * np.pi
    print(len(initial_params))

    maxiter=1000
    optimizer = COBYLA(maxiter=maxiter)

    cvar_histories = {}
    optimal_bitstrings = {}  # To store the best bitstring for each alpha
    optimal_values = {}  # To store the best objective value for each alpha

    # Optimization loop
    alphas = [0.25]
    objectives = []

    for alpha in alphas:
        print(f"Running optimization for alpha = {alpha}")
        # n, m, optimal_value, profits, weights, capacities = parse_mkp_dat_file(file_path)
        obj = Objective(H, offset, alpha, num_qubits, optimal_value)
        optimizer.minimize(fun=obj.evaluate, x0=initial_params)
        objectives.append(obj)
        
        # Store CVaR history for this alpha
        cvar_histories[alpha] = obj.history
        
        # Retrieve the optimal bitstring and objective value
        best_bitstring = None
        best_value = None
        
        # Iterate through the counts from the last iteration to find the best bitstring
        for bitstring, probability in obj.last_counts.items():
            value = eval_bitstring(H, bitstring) + offset
            if best_bitstring is None or value < best_value:
                best_bitstring = bitstring
                best_value = value
        
        # Convert bitstring to list of integers
        bitstring_as_list = [int(bit) for bit in best_bitstring]
        bitstring_as_list.reverse()
        print(f"Best Bitstring: {bitstring_as_list}")
        cost = qubo_shrunk.objective.evaluate(np.array(bitstring_as_list))
        print(f"Cost: {cost}")
        optimal_bitstrings[alpha] = bitstring_as_list
        optimal_values[alpha] = best_value

    

    # Print the optimal bitstrings and values for each alpha, and also feasibility
    print("\nOptimal Bitstrings and Objective Values:")
    for alpha in alphas:
        print(f"Alpha = {alpha}: Bitstring = {optimal_bitstrings[alpha]}, Objective Value = {optimal_values[alpha]}")
    # check if the solution is feasible
        

    # Save the results to a text filw in the 3_1_updated_results folder
    results_file = os.path.join(output_folder, "market_sharing_results_new.txt")
    with open(results_file, "w") as f:
        f.write("Optimal Bitstrings and Objective Values:\n")
        for alpha in alphas:
            f.write(f"Alpha = {alpha}: Bitstring = {optimal_bitstrings[alpha]}, Objective Value = {optimal_values[alpha]}\n")
        f.write("\nCVaR Histories:\n")
        for alpha, history in cvar_histories.items():
            f.write(f"Alpha = {alpha}: {history}\n")



    

    shrunk_solution = bitstring_as_list
    node_map = node_map

    # Map solution back to original nodes
    original_solution = map_solution_to_original_nodes(shrunk_solution, node_map)
    print("Solution for Original Graph:", original_solution)


    # shrunk_solution = xbest_brute_original
    shrunk_solution = original_solution
    print("Shrunk Solution:", shrunk_solution)


    full_solution = reconstruct_solution(shrinking_steps, shrunk_solution)
    print("Full Solution for Original Graph:", full_solution)


    # Example usage
    solution_dict = full_solution
    bitlist = solution_to_bitlist(solution_dict)

    print("Bitlist Representation:", bitlist)

    max_cut_utility = MaxCutUtility(max_cut_graph)
    qubo_bitstring = max_cut_utility.max_cut_to_qubo_solution(bitlist)
    qubo_cost = qubo.objective.evaluate(qubo_bitstring)
    print(f"QUBO cost: {qubo_cost}")

    # convert qubo to original problem
    problem_bitstring = converter.interpret(qubo_bitstring)
    feasibility = qp.get_feasibility_info(problem_bitstring)[0]
    print(f"Feasibility: {feasibility}")

    cost = qp.objective.evaluate(problem_bitstring)
    print(f"Cost: {cost}")
