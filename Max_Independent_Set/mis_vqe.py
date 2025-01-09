"""

Maximum Independent Set

"""

# basic imports
import os
import json 
import glob
import time
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
# quantum imports
import qiskit
print("Qiskit Version:",qiskit.__version__)
 
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
 
backend = AerSimulator(method='matrix_product_state')
 
 
 
estimator = BackendEstimatorV2(backend=backend)
sampler = BackendSamplerV2(backend=backend)


"""
Classical Helper Functions
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









import networkx as nx
import matplotlib.pyplot as plt

def find_simplicials(graph):
    """
    Finds simplicial nodes in the graph.
    A node is simplicial if its neighbors form a clique.

    Parameters:
    - graph: networkx.Graph, the input graph.

    Returns:
    - simplicials: list of simplicial nodes.
    """
    simplicials = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        is_simplicial = True
        # Check if all pairs of neighbors are connected
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if not graph.has_edge(neighbors[i], neighbors[j]):
                    is_simplicial = False
                    break
            if not is_simplicial:
                break
        if is_simplicial:
            simplicials.append(node)
    return simplicials


def preprocess_graph(graph):
    """
    Preprocesses the graph to iteratively remove simplicial nodes
    and their neighbors, and visualizes each step.

    Parameters:
    - graph: networkx.Graph, the input graph.

    Returns:
    - reduced_graph: networkx.Graph, the reduced graph after preprocessing.
    - simplicial_nodes: list of nodes fixed in the independent set.
    - original_indices: dict mapping reduced graph indices to original graph indices.
    """
    pos = nx.spring_layout(graph, seed=42)  # Position for consistent visualization
    iteration = 1
    simplicial_nodes = []  # To keep track of simplicial nodes added to the independent set

    while True:
        current_simplicial_nodes = find_simplicials(graph)
        if not current_simplicial_nodes:
            print(f"No more simplicial nodes found. Preprocessing complete.")
            break  # Stop if no simplicial nodes are found

        print(f"Iteration {iteration}: Found {len(current_simplicial_nodes)} simplicial nodes.")

        # Add simplicial nodes to the independent set
        simplicial_nodes.extend(current_simplicial_nodes)

        # Identify nodes to remove (simplicial nodes and their neighbors)
        neighbors_to_remove = set()
        for node in current_simplicial_nodes:
            neighbors_to_remove.update(graph.neighbors(node))
        nodes_to_remove = set(current_simplicial_nodes).union(neighbors_to_remove)

        # Visualize the current state of the graph
        # plt.figure(figsize=(10, 8))
        # nx.draw(graph, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10)
        # nx.draw_networkx_nodes(graph, pos, nodelist=current_simplicial_nodes, node_color="red", label="Simplicial Nodes")
        # nx.draw_networkx_nodes(graph, pos, nodelist=list(neighbors_to_remove), node_color="yellow", label="Neighbors of Simplicial Nodes")
        # plt.title(f"Iteration {iteration}: Removing Simplicial Nodes and Neighbors")
        # plt.legend()
        # plt.show()

        # Remove the nodes from the graph
        graph.remove_nodes_from(nodes_to_remove)
        iteration += 1

    print(f"Preprocessing complete. Remaining nodes: {list(graph.nodes())}")

    # Create a mapping from reduced graph indices to original indices
    original_indices = {i: node for i, node in enumerate(graph.nodes())}
    return graph, simplicial_nodes, original_indices


def combine_solutions(simplicial_nodes, reduced_solution, original_indices):
    """
    Combines the solution from the simplicial nodes and the reduced graph.

    Parameters:
    - simplicial_nodes: list of nodes fixed in the independent set during preprocessing.
    - reduced_solution: list of nodes in the independent set from the reduced graph solution.
    - original_indices: dict mapping reduced graph indices to original graph indices.

    Returns:
    - full_solution: list of nodes in the full independent set.
    """
    # Map reduced solution back to original graph indices
    mapped_solution = [original_indices[i] for i in reduced_solution]

    # Combine the simplicial nodes with the reduced graph solution
    full_solution = set(simplicial_nodes).union(mapped_solution)
    print(f"Simplicial nodes: {simplicial_nodes}")
    print(f"Reduced graph solution mapped to original indices: {mapped_solution}")
    return sorted(full_solution)


















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



# the file path
directory_path = "../mis_small_datasets/"
file_paths = glob.glob(os.path.join(directory_path, "*.txt"))

for file_path in file_paths:
    print("---------------------------------")
    print("---------------------------------")
    print(f"Processing file: {file_path}")

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

    

        

    # Check if the filename contains "tc"
    if "tc" in os.path.basename(file_path):
        print("Not doing preprocessing for this file.")
        num_nodes, edges = read_graph_from_file(file_path)
        model, x = create_max_independent_set_model(num_nodes, edges)
    
    else:
        # Preprocess the graph
        print("Preprocessing the graph.")
        G = nx.Graph()
        G.add_edges_from(edges)

        print(f"Initial graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")
        reduced_graph, simplicial_nodes, original_indices = preprocess_graph(G)
        print(f"Reduced graph has {len(reduced_graph.nodes)} nodes and {len(reduced_graph.edges)} edges.")
        reindexed_graph = nx.convert_node_labels_to_integers(reduced_graph)
        model, x = create_max_independent_set_model(len(reindexed_graph.nodes), list(reindexed_graph.edges))



    # making the reduced model 

    # G = nx.Graph()
    # G.add_edges_from(edges)

    # print(f"Initial graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")

    # # Preprocess the graph
    # reduced_graph, simplicial_nodes, original_indices = preprocess_graph(G)

    # print(f"Reduced graph has {len(reduced_graph.nodes)} nodes and {len(reduced_graph.edges)} edges.")


    # # Re-index the reduced graph
    # reindexed_graph = nx.convert_node_labels_to_integers(reduced_graph)
    # # Solve the reduced graph using the MIS model
    # model, x = create_max_independent_set_model(len(reindexed_graph.nodes), list(reindexed_graph.edges))
    



    qp = from_docplex_mp(model)
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)

    # number of variables
    num_vars = qubo.get_num_vars()
    print('Number of variables:', num_vars)
    print("---------------------------------")
    print("CVaR VQE")
    H, offset = qubo.to_ising()
    # make a quantum circuit to solve it 
    num_qubits = qubo.get_num_binary_vars()
    print(f"Number of qubits: {num_qubits}")

    reps = 10
    num_params = 2 * reps * num_qubits
    initial_params = np.random.rand(num_params) * 2 * np.pi

    print("The Depth of the quantum circuit is",reps,"and the number of parameters are",len(initial_params))


    maxiter=2000
    optimizer = COBYLA(maxiter=maxiter)#POWELL(maxfev=maxiter)

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


    # Initialize a dictionary to store results
    cvar_histories = {}
    optimal_bitstrings = {}  # To store the best bitstring for each alpha
    optimal_values = {}  # To store the best objective value for each alpha

    # Optimization loop
    start_time = time.time()
    alphas = [0.25]
    objectives = []
    optimal_value = optimal_value
    for alpha in alphas:
        print(f"Running optimization for alpha = {alpha}")
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
        bitstring_as_list = [int(bit) for bit in best_bitstring]
        
        optimal_bitstrings[alpha] = bitstring_as_list
        
        optimal_values[alpha] = best_value

    end_time = time.time()

    # Print the optimal bitstrings and values for each alpha, and also feasibility
    print("\nOptimal Bitstrings and Objective Values:")
    for alpha in alphas:
        # print(f"Alpha = {alpha}: Bitstring = {optimal_bitstrings[alpha]}, Objective Value = {optimal_values[alpha]}")
        
        # convert to market share solution 
        market_share_bitstring = converter.interpret(optimal_bitstrings[alpha])
        initial_feasible = qp.get_feasibility_info(market_share_bitstring)[0]
        market_share_cost = qp.objective.evaluate(market_share_bitstring)
        print(f"Feasible: {initial_feasible}")
        print(f"Max Independent Set: {market_share_cost}")
        print(f"Execution time: {end_time - start_time:.2f} seconds")

    # check if the solution is feasible
        

    output_folder = f"vqe_mdkp"  # Folder name based on num_products and seed
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    # Save the results to a text filw in the 3_1_updated_results folder
    results_file = os.path.join(output_folder, "mdkp_results_new.txt")
    with open(results_file, "w") as f:
        f.write("Optimal Bitstrings and Objective Values:\n")
        for alpha in alphas:
            # f.write(f"Alpha = {alpha}: Bitstring = {optimal_bitstrings[alpha]}, Objective Value = {optimal_values[alpha]}\n")
            f.write(f"Alpha = {alpha}: Bitstring = {converter.interpret(optimal_bitstrings[alpha])}, Objective Value = {qp.objective.evaluate(converter.interpret(optimal_bitstrings[alpha]))}\n")
        f.write("\nCVaR Histories:\n")
        for alpha, history in cvar_histories.items():
            f.write(f"Alpha = {alpha}: {history}\n")




    print("VQE Finished")
    print("-------------------------------------")


