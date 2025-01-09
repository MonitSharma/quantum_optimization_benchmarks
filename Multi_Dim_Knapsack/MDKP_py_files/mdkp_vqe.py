"""
Multi Dimensional Knapsack from the Dataset

Copyright: Monit Sharma @ SMU
"""
# necessary Imports
import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt 
from docplex.mp.model import Model



# quantum imports
import qiskit
print("Qiskit Version:",qiskit.__version__)
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms.optimizers import COBYLA,POWELL,SLSQP,P_BFGS
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
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


# file path for the desired data set
file_path = "../MKP_Instances/sac94/hp/hp1.dat"



"""
Since the benchmarking data is in a .dat file, we need to 
parse the data and formulate the problem out of it
"""

def parse_mkp_dat_file(file_path):
    """
    Parses a .dat file for the Multidimensional Knapsack Problem (MKP).

    Parameters:
    - file_path: str, path to the .dat file.

    Returns:
    - n: int, number of variables (items).
    - m: int, number of constraints (dimensions).
    - optimal_value: int, the optimal value (if available, otherwise 0).
    - profits: list of int, profit values for each item.
    - weights: 2D list of int, weights of items across constraints.
    - capacities: list of int, capacity values for each constraint.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Read the first line: n (variables), m (constraints), optimal value
    n, m, optimal_value = map(int, lines[0].strip().split())

    # Read the profits for each item
    profits = []
    i = 1
    while len(profits) < n:
        profits.extend(map(int, lines[i].strip().split()))
        i += 1

    # Read the weights (m x n matrix)
    weights = []
    for _ in range(m):
        weight_row = []
        while len(weight_row) < n:
            weight_row.extend(map(int, lines[i].strip().split()))
            i += 1
        weights.append(weight_row)

    # Read the capacities for each dimension
    capacities = []
    while len(capacities) < m:
        capacities.extend(map(int, lines[i].strip().split()))
        i += 1

    # Validate data dimensions
    if len(profits) != n:
        raise ValueError(f"Mismatch in number of items: Expected {n}, got {len(profits)}")
    for row in weights:
        if len(row) != n:
            raise ValueError(f"Mismatch in weights row length: Expected {n}, got {len(row)}")
    if len(capacities) != m:
        raise ValueError(f"Mismatch in number of capacities: Expected {m}, got {len(capacities)}")

    return n, m, optimal_value, profits, weights, capacities

def generate_mkp_instance(file_path):
    """
    Generates a Multidimensional Knapsack Problem (MKP) instance from a .dat file.

    Parameters:
    - file_path: str, path to the .dat file.

    Returns:
    - A dictionary containing the MKP instance details:
        - n: Number of items
        - m: Number of constraints
        - profits: Profit values for each item
        - weights: Weight matrix (m x n)
        - capacities: Capacities for each constraint
    """
    n, m, optimal_value, profits, weights, capacities = parse_mkp_dat_file(file_path)

    mkp_instance = {
        "n": n,
        "m": m,
        "optimal_value": optimal_value,
        "profits": profits,
        "weights": weights,
        "capacities": capacities
    }

    return mkp_instance

def print_mkp_instance(mkp_instance):
    """
    Prints the details of a Multidimensional Knapsack Problem (MKP) instance.

    Parameters:
    - mkp_instance: dict, the MKP instance details.
    """
    print(f"Number of items (n): {mkp_instance['n']}")
    print(f"Number of constraints (m): {mkp_instance['m']}")
    print(f"Optimal value (if known): {mkp_instance['optimal_value']}")
    # print("Profits:", mkp_instance['profits'])
    # print("Weights:")
    # for row in mkp_instance['weights']:
    #     print(row)
    # print("Capacities:", mkp_instance['capacities'])

def create_mkp_model(mkp_instance):
    """
    Creates a CPLEX model for the Multidimensional Knapsack Problem (MKP).

    Parameters:
    - mkp_instance: dict, the MKP instance details.

    Returns:
    - model: CPLEX model.
    - x: list of CPLEX binary variables representing item selection.
    """
    n = mkp_instance['n']
    m = mkp_instance['m']
    profits = mkp_instance['profits']
    weights = mkp_instance['weights']
    capacities = mkp_instance['capacities']

    # Create CPLEX model
    model = Model(name="Multidimensional Knapsack Problem")

    # Decision variables: x[i] = 1 if item i is selected, 0 otherwise
    x = model.binary_var_list(n, name="x")

    # Objective: Maximize total profit
    model.maximize(model.sum(profits[i] * x[i] for i in range(n)))

    # Constraints: Ensure total weights do not exceed capacity for each dimension
    for j in range(m):
        model.add_constraint(
            model.sum(weights[j][i] * x[i] for i in range(n)) <= capacities[j],
            f"capacity_constraint_{j}"
        )

    return model, x

# Example usage
if __name__ == "__main__":
    file_path = file_path # Replace with the path to your .dat file

    try:
        # Parse the .dat file and generate the MKP instance
        mkp_instance = generate_mkp_instance(file_path)

        # Print the MKP instance details
        print_mkp_instance(mkp_instance)

        # Create and solve the MKP model
        model, x = create_mkp_model(mkp_instance)

        # Solve the model
        """
        We don't need to classically solve the model, since the objective value
        is already given in the .dat file
        """
        # solution = model.solve()

        # if solution:
        #     print("Objective value (total profit):", solution.objective_value)
        #     selected_items = [i + 1 for i in range(mkp_instance['n']) if x[i].solution_value > 0.5]  # Convert to 1-based index
        #     print("Selected items:", selected_items)
        # else:
        #     print("No solution found.")
    except ValueError as e:
        print("Error in input data:", e)




"""
Convert it into a QUBO, since every quantum optimization method requires
the problem to be formulated as a QUBO
"""


qp = from_docplex_mp(model)                 # made a quadratic program (qp)
converter = QuadraticProgramToQubo()        # converter for qp to qubo  
qubo = converter.convert(qp)                # the qubo

num_vars = qubo.get_num_vars()
print("Number of Variables in QUBO:",num_vars)



"""
Let's Implement the VQE method first , we will be making use of the
CVaR VQE for three different values of alpha, and see which one performs the best

The ansatz used will be EfficientSU2, and we will be using SamplerV2 backends
"""

print("---------------------------------")
print("CVaR VQE implementation")


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


# Assigning Hyperparameters to the Quantum Circuit

reps = 4
num_params = 2 * reps * num_qubits
initial_params = np.random.rand(num_params) * 2 * np.pi

print("The Depth of the quantum circuit is",reps,"and the number of parameters are",len(initial_params))



optimizer = P_BFGS()



# Initialize a dictionary to store results
cvar_histories = {}
optimal_bitstrings = {}  # To store the best bitstring for each alpha
optimal_values = {}  # To store the best objective value for each alpha

# Optimization loop
alphas = [1.0, 0.25]
objectives = []
optimal_value = mkp_instance['optimal_value']
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
print("\nOptimal Bitstrings and Objective Values:")
for alpha in alphas:
    # print(f"Alpha = {alpha}: Bitstring = {optimal_bitstrings[alpha]}, Objective Value = {optimal_values[alpha]}")
    
    # convert to market share solution 
    market_share_bitstring = converter.interpret(optimal_bitstrings[alpha])
    initial_feasible = qp.get_feasibility_info(market_share_bitstring)
    market_share_cost = qp.objective.evaluate(market_share_bitstring)
    print(f"Feasible: {initial_feasible}")
    print(f"Multi Dimension Knapsack cost: {market_share_cost}")

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
