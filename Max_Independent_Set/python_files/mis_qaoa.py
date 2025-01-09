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


# the file path
file_path = "../mis_datasets/1dc.64.txt"


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
 

# pce imports

sys.path.append(os.path.abspath(os.path.join('../..')))
from pce_qubo.pauli_correlation_encoding import PauliCorrelationEncoding
from pce_qubo.mixed_optimize import PauliCorrelationOptimizer

from pce_qubo.utility import Utility
from qiskit.quantum_info import SparsePauliOp, Statevector



# function to read the file


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
 
 
"""
Let's Implement the VQE method first , we will be making use of the
CVaR QAOA for three different values of alpha, and see which one performs the best
 
The ansatz used will be QAOAAnsatz, and we will be using SamplerV2 backends
"""
 
print("---------------------------------")
print("CVaR QAOA implementation")
 
 
# get the Hamiltonian from the QUBO
 
H, offset = qubo.to_ising()
 
# how many number of qubits, will be requried??
# make a quantum circuit to solve it
num_qubits = qubo.get_num_binary_vars()
print(f"Number of qubits: {num_qubits}")
 
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
 
 
 
 
# function to make qaoa circuit
def generate_sum_x_pauli_str(length):
    ret = []
    for i in range(length):
        paulis = ['I'] * length
        paulis[i] = 'X'
        ret.append(''.join(paulis))
 
    return ret
 
def qaoa_circuit(problem_ham: SparsePauliOp, depth: int = 1) -> QuantumCircuit:
    r"""
    Input:
    - problem_ham: Problem Hamiltonian to construct the QAOA circuit.
    Standard procedure would be:
    ```
        hamiltonian, offset = qubo.to_ising()
        qc = qaoa_circuit_from_qubo(hamiltonian)
    ```
 
    Returns:
    - qc: A QuantumCircuit object representing the QAOA circuit e^{-i\beta H_M} e^{-i\gamma H_C}.
    """
    num_qubits = problem_ham.num_qubits
 
    gamma = ParameterVector(name=r'$\gamma$', length=depth)
    beta = ParameterVector(name=r'$\beta$', length=depth)
 
    mixer_ham = SparsePauliOp(generate_sum_x_pauli_str(num_qubits))
   
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
 
    for p in range(depth):
        exp_gamma = PauliEvolutionGate(problem_ham, time=gamma[p])
        exp_beta = PauliEvolutionGate(mixer_ham, time=beta[p])
        qc.append(exp_gamma, qargs=range(num_qubits))
        qc.append(exp_beta, qargs=range(num_qubits))
 
    return qc
 
# objective function for the gradient to optimize
 
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
        # Create a new qaoa circuit
        qc = qaoa_circuit(self.H, depth=reps)
 
        qc = qc.decompose()
 
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
 
 
# Assigning Hyperparameters to the Quantum Circuit
 
reps = 1
num_params = 2 * reps
initial_params = np.random.rand(num_params) * 2 * np.pi
 
print("The Depth of the quantum circuit is",reps,"and the number of parameters are",len(initial_params))
 
 
maxiter = 1
optimizer = COBYLA(1)
 
 
 
# Initialize a dictionary to store results
cvar_histories = {}
optimal_bitstrings = {}  # To store the best bitstring for each alpha
optimal_values = {}  # To store the best objective value for each alpha
 
# Optimization loop
alphas = [1.0]
objectives = []
optimal_value = 10
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
    optimal_bitstrings[alpha] = best_bitstring
    optimal_values[alpha] = best_value
 
# Plotting the CVaR histories
plt.figure(figsize=(10, 6))
for alpha, history in cvar_histories.items():
    plt.plot(history, label=f'Alpha = {alpha}')
 
plt.xlabel('Iteration')
plt.ylabel('CVaR')
plt.title('CVaR History for Different Values of Alpha')
plt.legend()
plt.grid(True)
plt.show()
 
# Print the optimal bitstrings and values for each alpha, and also feasibility
# Print the optimal bitstrings and values for each alpha, and also feasibility
print("\nOptimal Bitstrings and Objective Values:")
for alpha in alphas:
    # print(f"Alpha = {alpha}: Bitstring = {optimal_bitstrings[alpha]}, Objective Value = {optimal_values[alpha]}")
    print(len(optimal_bitstrings[alpha]))

    bitstring_as_list = [int(bit) for bit in optimal_bitstrings[alpha]]
    # convert to market share solution
    market_share_bitstring = converter.interpret(bitstring_as_list)
    initial_feasible = qp.get_feasibility_info(market_share_bitstring)[0]
    market_share_cost = qp.objective.evaluate(market_share_bitstring)
    print(f"Feasible: {initial_feasible}")
    print(f"Maximum Independent Set: {market_share_cost}")
 
# check if the solution is feasible
   
 
output_folder = f"qaoa_mis"  # Folder name based on num_products and seed
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
# Save the results to a text filw in the 3_1_updated_results folder
results_file = os.path.join(output_folder, "mis_results_qaoa.txt")
with open(results_file, "w") as f:
    f.write("Optimal Bitstrings and Objective Values:\n")
    for alpha in alphas:
        # Convert the binary string to a list of integers
        bitstring_as_list = [int(bit) for bit in optimal_bitstrings[alpha]]
        
        # Interpret the solution and evaluate the QUBO cost
        interpreted_bitstring = converter.interpret(bitstring_as_list)
        objective_value = qp.objective.evaluate(interpreted_bitstring)
        
        f.write(f"Alpha = {alpha}: Bitstring = {interpreted_bitstring}, Objective Value = {objective_value}\n")
    
    f.write("\nCVaR Histories:\n")
    for alpha, history in cvar_histories.items():
        f.write(f"Alpha = {alpha}: {history}\n")
 
 
 