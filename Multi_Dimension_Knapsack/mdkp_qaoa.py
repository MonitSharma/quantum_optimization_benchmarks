import time 
print("QAOA Start")
full_start_time = time.time()
from docplex.mp.model import Model
from qiskit.circuit.library import QAOAAnsatz
import qiskit 
print("Qiskit Version: ",qiskit.__version__)
import os
from datetime import datetime
import glob
# basic imports
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# quantum imports
from qiskit_optimization.applications import Maxcut, Knapsack
from qiskit.circuit import Parameter,QuantumCircuit
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# SciPy minimizer routine
from scipy.optimize import minimize
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit_aer import AerSimulator
backend = AerSimulator(method='matrix_product_state')


estimator = BackendEstimatorV2(backend=backend)
sampler = BackendSamplerV2(backend=backend, options={"default_shots": 8000})


"""
Create your qp instance here, till the qubo making part
"""


def parse_mkp_dat_file(file_path):
        """
        Parses a .dat file for the Multidimensional Knapsack qp (MKP).

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
    Generates a Multidimensional Knapsack qp (MKP) instance from a .dat file.

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
    Prints the details of a Multidimensional Knapsack qp (MKP) instance.

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
    Creates a CPLEX model for the Multidimensional Knapsack qp (MKP).

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
    model = Model(name="Multidimensional Knapsack qp")

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


directory_path = "MKP_Instances/hpp/"

# Get all .dat files in the directory
file_paths = glob.glob(os.path.join(directory_path, "*.dat"))


for file_path in file_paths:
    
    print("---------------------------------")
    print("---------------------------------")
    print(f"Processing file: {file_path}")

    mkp_instance = generate_mkp_instance(file_path)

    # Print the MKP instance details
    print_mkp_instance(mkp_instance)

    # Create and solve the MKP model
    model, x = create_mkp_model(mkp_instance)

    print()

    qp = from_docplex_mp(model)                 # made a quadratic program (qp)
    converter = QuadraticProgramToQubo()        # converter for qp to qubo  
    qubo = converter.convert(qp)                # the qubo



    num_vars = qubo.get_num_vars()
    print('Number of variables:', num_vars)
    reps = 5
    print('Number of repetitions:', reps)
    # converting hamiltonian
    qubitOp, offset = qubo.to_ising()

    ## QAOA

    

    
    circuit = QAOAAnsatz(cost_operator=qubitOp, reps=reps)
    #circuit.measure_all()
    circuit = circuit.decompose(reps=3)
    circuit.draw('mpl',fold=-1)

    initial_gamma = np.pi
    initial_beta = np.pi/2

    init_params = [initial_gamma, initial_beta]*reps

    print("Number of parameters:", len(init_params))
    # number of qubits, circuit depth, gate counts, 2 qubit gate count
    print('Number of qubits:', circuit.num_qubits)
    print('Circuit depth:', circuit.depth())
    print('Gate counts:', dict(circuit.count_ops()))
    # print new line
    print()
    print("-----------------------------------------------------")

    # cost function

    cost_history_dict = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }

    def cost_func_estimator(params, ansatz, hamiltonian, estimator):

        # transform the observable defined on virtual qubits to
        # an observable defined on all physical qubits
        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])

        results = job.result()[0]
        cost = results.data.evs


        cost_history_dict["iters"] += 1
        cost_history_dict["prev_vector"] = params
        cost_history_dict["cost_history"].append(cost)
        print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {cost}]")



        objective_func_vals.append(cost)


        return cost



    objective_func_vals = [] # Store the objective function values
    optimization_time_start = time.time()
    result = minimize(
        cost_func_estimator,
        init_params,
        args= (circuit, qubitOp, estimator),
        method="Powell",
        tol=1e-3
    )
    optimization_time_end = time.time()

    print(result)

    output_folder = "output_plots"
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"cost_vs_iterations_{timestamp}.png"
    output_path = os.path.join(output_folder, file_name)

    # Plot "Cost vs. Iteration"
    plt.figure(figsize=(12, 6))
    plt.plot(objective_func_vals, label="Objective Function")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost vs. Iteration")
    plt.legend()

    # Save the plot
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()  # Close the plot to free up memory

    print(f"Plot saved to: {output_path}")

    # post processing
    post_processing_time_start = time.time()
    optimized_circuit = circuit.assign_parameters(result.x)
    optimized_circuit.measure_all()
    optimized_circuit.draw('mpl', idle_wires=False,fold=-1)

    pub = (optimized_circuit,)
    job = sampler.run([pub], shots=int(1e4))
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_int.values())
    final_distribution_int = {key: val/shots for key, val in counts_int.items()}
    final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}

    post_processing_time_end = time.time()

    def to_bitstring(integer, num_bits):
        result = np.binary_repr(integer, width=num_bits)
        return [int(digit) for digit in result]

    keys = list(final_distribution_int.keys())
    values = list(final_distribution_int.values())
    most_likely = keys[np.argmax(np.abs(values))]
    most_likely_bitstring = to_bitstring(most_likely, num_vars)
    most_likely_bitstring.reverse()

    print("Result bitstring:", most_likely_bitstring)




    # Find the indices of the top 4 values
    top_4_indices = np.argsort(np.abs(values))[::-1][:4]
    top_4_results = []
    # Print the top 4 results with their probabilities
    print("Top 4 Results:")
    for idx in top_4_indices:
        bitstring = to_bitstring(keys[idx], num_vars)
        bitstring.reverse()
        top_4_results.append(bitstring)
        print(f"Bitstring: {bitstring}, Probability: {values[idx]:.6f}")



    # Update matplotlib font size
    matplotlib.rcParams.update({"font.size": 10})

    # Assuming final_distribution_bin is defined elsewhere
    final_bits = final_distribution_bin  

    # Get the absolute values and sort to extract the top 16 and top 4 values
    values = np.abs(list(final_bits.values()))
    top_16_values = sorted(values, reverse=True)[:16]
    top_4_values = sorted(values, reverse=True)[:10]

    # Filter the top 16 bitstrings and their probabilities
    top_16_bitstrings = []
    top_16_probabilities = []

    for bitstring, value in final_bits.items():
        if abs(value) in top_16_values:
            top_16_bitstrings.append(bitstring)
            top_16_probabilities.append(value)

    # Sort the top 16 by probability for better visualization
    sorted_indices = np.argsort(top_16_probabilities)[::-1]
    top_16_bitstrings = [top_16_bitstrings[i] for i in sorted_indices]
    top_16_probabilities = [top_16_probabilities[i] for i in sorted_indices]

    file_name = f"result_distribution_{timestamp}.png"
    output_path = os.path.join(output_folder, file_name)
    # Plot the top 16 values
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks(rotation=45)
    plt.title("Result Distribution")
    plt.xlabel("Bitstrings (reversed)")
    plt.ylabel("Probability")

    bars = ax.bar(top_16_bitstrings, top_16_probabilities, color="tab:grey")

    # Highlight the top 4 bars in purple
    for i, bar in enumerate(bars):
        if top_16_probabilities[i] in top_4_values:
            bar.set_color("tab:purple")

    # Save the plot
    plt.savefig(output_path, format="png", dpi=300)
    plt.close(fig)  # Close the figure to free up memory

    print(f"Plot saved to: {output_path}")


    # convert 


    result = converter.interpret(most_likely_bitstring)
    cost = qp.objective.evaluate(result)
    feasible =qp.get_feasibility_info(result)[0]


    # print("Result knapsack:", result)
    # print("Result value:", cost)
    # print("Feasible:", feasible)

    # Iterate through the list of bitstrings and evaluate for each
    for bitstring in top_4_results:
        result = converter.interpret(bitstring)  # Interpret the bitstring
        cost = qp.objective.evaluate(result)  # Evaluate the cost for the bitstring
        feasible =qp.get_feasibility_info(result)[0]
        
        # Print the results
        print("Result knapsack:", result)
        print("Result value:", cost)
        print("Feasible solution:", feasible)
        print("--------------------")


    full_end_time = time.time()

    print()
    print("-----------------------------------------------------")
    print("Time taken for optimization:", optimization_time_end - optimization_time_start, "seconds")
    print("Time taken for post-processing:", post_processing_time_end - post_processing_time_start, "seconds")
    print("Total time taken:", full_end_time - full_start_time, "seconds")
    # use this to run
    # python -u .\qaoa.py > qaoa_out.log 2>&1

