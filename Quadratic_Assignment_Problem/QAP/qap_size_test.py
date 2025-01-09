# necessary files and basic functions
import glob
import os
import numpy as np
from docplex.mp.model import Model


# quantum imports
import os
import json
import numpy as np
from docplex.mp.model import Model
import io
import time
import os
import json
import sys
import numpy as np
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms.optimizers import ADAM, POWELL, SLSQP, COBYLA, P_BFGS





### Quantum imports
import qiskit
from qiskit.compiler import transpile
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA,POWELL,SLSQP,L_BFGS_B, P_BFGS
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2

backend = AerSimulator(method='matrix_product_state')



estimator = BackendEstimatorV2(backend=backend)
sampler = BackendSamplerV2(backend=backend)
from qiskit.quantum_info import SparsePauliOp, Statevector


def parse_qap_dat_file(file_path):
    """
    Parses a .dat file for the Quadratic Assignment Problem (QAP).

    Parameters:
    - file_path: str, path to the .dat file.

    Returns:
    - n: int, the size of the problem (number of facilities/locations).
    - A: 2D numpy array, flow matrix.
    - B: 2D numpy array, distance matrix.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Read the size of the problem
    n = int(lines[0].strip())

    # Read the flow matrix (A)
    A = []
    current_line = 1
    for i in range(n):
        row = list(map(int, lines[current_line].strip().split()))
        while len(row) < n:  # Handle cases where rows are split across multiple lines
            current_line += 1
            row.extend(list(map(int, lines[current_line].strip().split())))
        A.append(row[:n])
        current_line += 1

    # Read the distance matrix (B)
    B = []
    for i in range(n):
        row = list(map(int, lines[current_line].strip().split()))
        while len(row) < n:  # Handle cases where rows are split across multiple lines
            current_line += 1
            row.extend(list(map(int, lines[current_line].strip().split())))
        B.append(row[:n])
        current_line += 1

    return n, np.array(A), np.array(B)

def create_qap_model(n, A, B):
    """
    Creates a CPLEX model for the Quadratic Assignment Problem (QAP).

    Parameters:
    - n: int, the size of the problem (number of facilities/locations).
    - A: 2D numpy array, flow matrix.
    - B: 2D numpy array, distance matrix.

    Returns:
    - model: CPLEX model.
    - x: 2D list of CPLEX binary variables representing the assignment.
    """
    # Create a CPLEX model
    model = Model(name="Quadratic Assignment Problem")

    # Decision variables: x[i][j] = 1 if facility i is assigned to location j
    x = [[model.binary_var(name=f"x_{i}_{j}") for j in range(n)] for i in range(n)]

    # Objective: Minimize the total cost
    model.minimize(
        model.sum(A[i, k] * B[j, l] * x[i][j] * x[k][l]
                  for i in range(n) for j in range(n) for k in range(n) for l in range(n))
    )

    # Constraints: Each facility is assigned to exactly one location
    for i in range(n):
        model.add_constraint(model.sum(x[i][j] for j in range(n)) == 1, f"facility_assignment_{i}")

    # Constraints: Each location is assigned exactly one facility
    for j in range(n):
        model.add_constraint(model.sum(x[i][j] for i in range(n)) == 1, f"location_assignment_{j}")

    return model, x


    
    

directory_path = "../qap_test/"           # "../qap_small_data/"

file_paths = glob.glob(os.path.join(directory_path, "*.dat"))
for file_path in file_paths:
    print(f"Processing file: {file_path}")

    file_path = file_path  # Replace with the path to your .dat file

    # Parse the .dat file
    n, A, B = parse_qap_dat_file(file_path)

    # Create the QAP model
    model, x = create_qap_model(n, A, B)
    #print(model.export_to_string())
    # print number of variables and constraints
    print("Number of variables: ", model.number_of_variables)

    # Solve the model
    solution = model.solve()

    if solution:
        print("Objective value:", solution.objective_value)
        assignment = [[x[i][j].solution_value for j in range(n)] for i in range(n)]
        print("Assignment matrix:")
        for row in assignment:
            print(row)
    else:
        print("No solution found.")

    optimal_value = solution.objective_value


    # quadratic model
    qp = from_docplex_mp(model)
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)
    num_vars = qubo.get_num_vars()
    print('Number of variables:', num_vars)

    # Hamiltonian and Offset
    H, offset = qubo.to_ising()

    # how many qubits are required??
    num_qubits = qubo.get_num_binary_vars()
    print(f"Number of qubits: {num_qubits}")


    # function to compute CVAR

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


    # Function to evaluate the bitstring

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


    # objective function for the optimizer
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
            
            # transpile(qc,backend)

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


    reps = 1
    num_params = 2 * reps * num_qubits
    initial_params = np.random.rand(num_params) * 2 * np.pi

    print("The depth of quantum circuit is",reps,"and the number of variational parameters are ",len(initial_params))


    maxiter=1
    optimizer = COBYLA(maxiter=maxiter)



    # Initialize a dictionary to store results
    cvar_histories = {}
    optimal_bitstrings = {}  # To store the best bitstring for each alpha
    optimal_values = {}  # To store the best objective value for each alpha

    # Optimization loop
    alphas = [0.25]
    objectives = []

    for alpha in alphas:
        print(f"Running optimization for alpha = {alpha}")
        obj = Objective(H, offset, alpha, num_qubits, optimal_value=None)
        
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
        
        optimal_bitstrings[alpha] = bitstring_as_list
        optimal_values[alpha] = best_value


    # Print the optimal bitstrings and values for each alpha, and also feasibility
    print("\nOptimal Bitstrings and Objective Values:")
    for alpha in alphas:
        print(f"Alpha = {alpha}: Bitstring = {optimal_bitstrings[alpha]}, Objective Value = {optimal_values[alpha]}")
        # convert to market share solution 
        market_share_bitstring = converter.interpret(optimal_bitstrings[alpha])
        initial_feasible = qp.get_feasibility_info(market_share_bitstring)[0]
        market_share_cost = qp.objective.evaluate(market_share_bitstring)

        print(f"Feasible: {initial_feasible}")
        print(f"Market share cost: {market_share_cost}")




    