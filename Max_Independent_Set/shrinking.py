import cvxpy as cp
import numpy as np
import rustworkx as rx
from docplex.mp.model import Model
import matplotlib.pyplot as plt

def create_graph_from_weight_matrix(w):
    """
    Create a Rustworkx graph from a weight matrix.

    Parameters:
        w (np.array): Weight matrix representing the graph.

    Returns:
        rustworkx.PyGraph: Graph created from the weight matrix.
    """
    G = rx.PyGraph()
    n = len(w)

    # Add nodes
    G.add_nodes_from(range(n))

    # Add edges with weights, ignoring zero-weight edges
    for i in range(n):
        for j in range(i + 1, n):
            if w[i, j] != 0:
                G.add_edge(i, j, w[i, j])

    return G


def draw_graph_rustworkx(G, colors=None):
    """
    Visualize a Rustworkx graph using Matplotlib.

    Parameters:
        G (rustworkx.PyGraph): Graph to be visualized.
        colors (list): List of colors for nodes (optional).
    """
    # Convert Rustworkx graph to NetworkX format for visualization
    import networkx as nx
    nx_graph = nx.Graph()
    for index in G.node_indexes():
        nx_graph.add_node(index)

    for edge in G.weighted_edge_list():
        u, v, weight = edge
        nx_graph.add_edge(u, v, weight=weight)

    # Get layout for nodes
    pos = nx.spring_layout(nx_graph, seed=42)

    # Set default colors if not provided
    if colors is None:
        colors = ["lightblue"] * len(G.node_indexes())

    # Draw the graph
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(nx_graph, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(nx_graph, "weight")
    nx.draw_networkx_edge_labels(nx_graph, pos=pos, edge_labels=edge_labels)

    plt.show()

def solve_weighted_maxcut_cplex(graph):
    """
    Solve the Weighted Max-Cut problem using CPLEX.

    Parameters:
        graph (rustworkx.PyGraph): Input graph with weighted edges.

    Returns:
        partition_assignments (dict): Dictionary mapping nodes to partitions (0 or 1).
        cut_value (float): Total cut value of the solution.
    """
    # Step 1: Initialize CPLEX model
    mdl = Model(name="Weighted Max-Cut")

    # Step 2: Define binary variables for each node
    nodes = list(graph.node_indexes())
    y = mdl.binary_var_dict(nodes, name="y")

    # Step 3: Define the objective function
    objective = mdl.sum(
        graph.get_edge_data(u, v) * (y[u] + y[v] - 2 * y[u] * y[v])
        for u, v, _ in graph.weighted_edge_list()
    )
    mdl.maximize(objective)

    # Step 4: Solve the problem
    solution = mdl.solve(log_output=True)
    if not solution:
        raise ValueError("No solution found by CPLEX.")

    # Step 5: Extract partition assignments
    partition_assignments = {node: int(solution[y[node]]) for node in nodes}

    # Step 6: Compute total cut value
    cut_value = 0
    crossing_edges = []
    for u, v, weight in graph.weighted_edge_list():
        if partition_assignments[u] != partition_assignments[v]:
            cut_value += weight
            crossing_edges.append((u, v))

    return partition_assignments, cut_value, crossing_edges


# Visualize the solution
def visualize_maxcut_solution(graph, partition_assignments, crossing_edges, cut_value):
    """
    Visualize the Max-Cut solution based on the final partition assignments.

    Parameters:
        graph (rustworkx.PyGraph): The original graph.
        partition_assignments (dict): Dictionary mapping node IDs to their partition (0 or 1).
        crossing_edges (list): List of edges crossing the cut.
        cut_value (float): Total cut value.
        
    Returns:
        None: Displays the Max-Cut visualization.
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    # Convert rustworkx graph to networkx graph for visualization
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(graph.node_indexes())
    for u, v, weight in graph.weighted_edge_list():
        nx_graph.add_edge(u, v, weight=weight)

    # Separate nodes into partitions
    partition_0 = [node for node, part in partition_assignments.items() if part == 0]
    partition_1 = [node for node, part in partition_assignments.items() if part == 1]

    # Plot the graph
    pos = nx.spring_layout(nx_graph, seed=42)  # Layout for consistent visualization
    plt.figure(figsize=(10, 8))

    # Draw nodes with different colors for the two partitions
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=partition_0, node_color="lightblue", label="Partition 0", node_size=600)
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=partition_1, node_color="orange", label="Partition 1", node_size=600)

    # Draw all edges
    nx.draw_networkx_edges(nx_graph, pos, edge_color="gray", alpha=0.5, width=1)

    # Highlight crossing edges
    nx.draw_networkx_edges(nx_graph, pos, edgelist=crossing_edges, edge_color="red", width=2, label="Crossing Edges")

    # Add edge labels
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in nx_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels)

    # Add labels to nodes
    nx.draw_networkx_labels(nx_graph, pos, font_size=12, font_color="black")

    # Title and legend
    plt.title(f"Max-Cut Solution Visualization (Cut Value: {cut_value})")
    plt.legend(loc="best")
    plt.show()

    # Print the solution details
    print("Partition 0 (Blue):", partition_0)
    print("Partition 1 (Orange):", partition_1)
    print("Edges Crossing the Cut:", crossing_edges)
    print("Total Cut Value:", cut_value)


def solution_to_bitlist(solution_dict):
    """
    Transform a dictionary solution for a Weighted Max-Cut into a bitlist.

    Parameters:
        solution_dict (dict): The solution dictionary, where keys are node indices and values are partition assignments (0 or 1).

    Returns:
        list: A bitlist representation of the solution.
    """
    # Ensure the dictionary is processed in sorted order of keys
    sorted_keys = sorted(solution_dict.keys())
    bitlist = [solution_dict[key] for key in sorted_keys]
    return bitlist


"""
Max Cut SDP Relaxation
"""

def max_cut_sdp(graph):
    """
    Solves the Max-Cut problem using SDP relaxation.
    """
    # Map graph nodes to contiguous indices
    current_nodes = list(graph.node_indexes())
    node_map = {node: idx for idx, node in enumerate(current_nodes)}

    # Step 1: Construct the Laplacian matrix dynamically
    n = len(current_nodes)
    L = np.zeros((n, n))
    for edge in graph.edge_list():
        i, j = edge
        weight = graph.get_edge_data(i, j)
        i_mapped = node_map[i]
        j_mapped = node_map[j]
        L[i_mapped, i_mapped] += weight
        L[j_mapped, j_mapped] += weight
        L[i_mapped, j_mapped] -= weight
        L[j_mapped, i_mapped] -= weight

    # Normalize the Laplacian to avoid numerical issues
    L = L / np.max(np.abs(L))

    # Step 2: Define the SDP problem
    X = cp.Variable((n, n), symmetric=True)
    objective = cp.Maximize(cp.trace(L @ X) / 4)
    constraints = [X >> np.eye(n) * 1e-6, cp.diag(X) == 1]  # Add regularization

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, max_iters=500000, eps=1e-9, verbose=False)

    print("Solver Status:", prob.status)
    print("Objective Value:", prob.value)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("Warning: SDP solver did not find an optimal solution.")
        return None, None

    # Step 3: Retrieve and analyze the solution
    X_opt = X.value

    # Step 4: Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(X_opt)
    largest_eigenvalue = np.max(eigenvalues)
    threshold = largest_eigenvalue * 1e-6
    filtered_eigenvalues = np.where(eigenvalues > threshold, eigenvalues, 0)

    # Step 5: Decompose into B matrix
    B = eigenvectors @ np.diag(np.sqrt(filtered_eigenvalues))
    B_normalized = B / np.linalg.norm(B, axis=1, keepdims=True)

    return B_normalized, X_opt

def calculate_sdp_correlations(B):
    """
    Calculates SDP correlations from normalized matrix B.
    
    Parameters:
        B (np.array): Decomposed and normalized SDP solution matrix.
        
    Returns:
        np.array: Correlation matrix.
    """
    n = B.shape[0]
    correlations = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            correlations[i, j] = np.dot(B[i], B[j])  # Dot product between rows
            correlations[j, i] = correlations[i, j]
    #print("Computed Correlations:\n", correlations)
    return correlations


import matplotlib.pyplot as plt
import networkx as nx

def plot_graph_step(graph, title, partition=None):
    """
    Visualize the current state of the graph with optional partition coloring.
    
    Parameters:
        graph (rustworkx.PyGraph): The graph to visualize.
        title (str): Title for the plot.
        partition (dict): Dictionary mapping nodes to partitions (optional).
    """
    # Convert graph to networkx for visualization
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(graph.node_indexes())
    for u, v, weight in graph.weighted_edge_list():
        nx_graph.add_edge(u, v, weight=weight)
    
    # Assign colors based on partitions
    if partition:
        colors = [partition[node] if partition[node] is not None else -1 for node in nx_graph.nodes()]
    else:
        colors = "lightblue"
    
    # Get layout
    pos = nx.spring_layout(nx_graph, seed=42)
    
    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(
        nx_graph, pos, with_labels=True, node_color=colors, cmap=plt.cm.Paired, node_size=500, edge_color="gray"
    )
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in nx_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.show()


from sklearn.cluster import KMeans

def determine_threshold(correlations, method="clustering", alpha=0.8, step=None, initial_threshold=0.8, decay_rate=0.9):
    """
    Determine the threshold for merging based on the correlation matrix.
    
    Parameters:
        correlations (np.array): Correlation matrix.
        method (str): Method to compute the threshold. Options:
            - "proportional": Proportional to maximum correlation.
            - "mean": Mean of absolute correlations.
            - "median": Median of absolute correlations.
            - "quantile": Quantile-based threshold (default 90th percentile).
            - "clustering": Clustering-based threshold using KMeans.
            - "decay": Exponential decay threshold.
        alpha (float): Scaling factor or quantile (if applicable).
        step (int): Current shrinking step (used for decay method).
        initial_threshold (float): Starting threshold for decay method.
        decay_rate (float): Decay rate for the threshold.
        
    Returns:
        float: Computed threshold value.
    """
    abs_correlations = np.abs(correlations)
    # Flatten the upper triangular part for clustering and percentile methods
    flattened_correlations = abs_correlations[np.triu_indices_from(abs_correlations, k=1)]

    if method == "proportional":
        return alpha * np.max(abs_correlations)  # Relative to max correlation
    elif method == "mean":
        return np.mean(abs_correlations)  # Mean correlation
    elif method == "median":
        return np.median(abs_correlations)  # Median correlation
    elif method == "quantile":
        return np.quantile(flattened_correlations, alpha)  # Quantile-based
    elif method == "clustering":
        kmeans = KMeans(n_clusters=2, random_state=42).fit(flattened_correlations.reshape(-1, 1))
        return np.mean(kmeans.cluster_centers_)  # Average of cluster centers
    elif method == "decay":
        if step is None:
            raise ValueError("For decay method, 'step' must be provided.")
        return initial_threshold * (decay_rate ** step)  # Exponential decay
    else:
        raise ValueError(f"Unknown method: {method}")



def shrink_graph_with_tracking(graph, correlations, target_num_nodes, recalculation_steps=5):
    """
    Shrink the graph while tracking the merging steps for reconstruction.

    Parameters:
        graph (rustworkx.PyGraph): The input graph to shrink.
        correlations (np.array): Initial correlation matrix from SDP.
        target_num_nodes (int): Desired number of nodes to stop shrinking at.
        threshold (float): Threshold for merging nodes.
        recalculation_steps (int): Number of steps between correlation matrix recalculations.

    Returns:
        shrunk_graph (rustworkx.PyGraph): The reduced graph with the desired number of nodes.
        partition (dict): Mapping of nodes to partitions.
        shrinking_steps (list of tuples): List of merging steps for reconstruction.
    """
    shrunk_graph = graph.copy()
    partition = {node: None for node in shrunk_graph.node_indexes()}
    current_mapping = {node: idx for idx, node in enumerate(shrunk_graph.node_indexes())}  # Map node IDs to correlation indices
    shrinking_steps = []  # List to track merging steps
    step = 1  # Step counter for visualization
    steps_since_recalculation = 0  # Counter for recalculation frequency

    # Continue shrinking until the graph has the desired number of nodes
    while shrunk_graph.num_nodes() > target_num_nodes:
        # Visualization before the shrinking step
        # plot_graph_step(shrunk_graph, f"Step {step}: Before Shrinking", partition)

        # Align correlations with current graph structure
        current_nodes = list(current_mapping.keys())
        correlations = correlations[np.ix_(list(current_mapping.values()), list(current_mapping.values()))]

        # Dynamically determine the threshold
        threshold = determine_threshold(correlations, method="proportional", alpha=0.8)
        print(f"Dynamic Threshold (Step {step}): {threshold}")

        max_correlation = 0
        max_pair = None

        # Find the edge with the largest absolute correlation
        for i in range(len(current_nodes)):
            for j in range(i + 1, len(current_nodes)):
                if abs(correlations[i, j]) > max_correlation:
                    max_correlation = abs(correlations[i, j])
                    max_pair = (current_nodes[i], current_nodes[j])

        if max_pair is None:
            print("No correlations above threshold found.")
            break

        i, j = max_pair
        sigma_ij = np.sign(correlations[current_mapping[i], current_mapping[j]])

        # Record the merging step
        shrinking_steps.append((i, j, sigma_ij, j))  # (node_i, node_j, sigma_ij, merged_into)

        # Update the partition information
        if sigma_ij > 0:
            partition[i] = partition[j] = 0 if partition[i] is None else partition[i]
        else:
            if partition[i] is None and partition[j] is None:
                partition[i], partition[j] = 0, 1
            elif partition[i] is None:
                partition[i] = 1 - partition[j]
            elif partition[j] is None:
                partition[j] = 1 - partition[i]

        # Update the graph by merging nodes i and j
        if shrunk_graph.has_edge(i, j):
            weight_ij = shrunk_graph.get_edge_data(i, j)
            shrunk_graph.remove_edge(i, j)
        else:
            weight_ij = 0

        for neighbor in shrunk_graph.neighbors(i):
            if neighbor != j:
                weight_ik = shrunk_graph.get_edge_data(i, neighbor)
                if shrunk_graph.has_edge(j, neighbor):
                    weight_jk = shrunk_graph.get_edge_data(j, neighbor)
                    new_weight = weight_jk + sigma_ij * weight_ik
                    shrunk_graph.update_edge(j, neighbor, new_weight)
                else:
                    new_weight = sigma_ij * weight_ik
                    shrunk_graph.add_edge(j, neighbor, new_weight)

        # Remove node i from the graph
        shrunk_graph.remove_node(i)

        # Update the mapping to reflect the removed node
        del current_mapping[i]
        current_mapping = {node: idx for idx, node in enumerate(shrunk_graph.node_indexes())}  # Remap node IDs to correlation indices

        # Increment the step counter
        step += 1
        steps_since_recalculation += 1

        # Recalculate correlations after the specified number of steps
        if steps_since_recalculation >= recalculation_steps:
            print(f"Recalculating correlations at step {step}.")
            B, X_opt = max_cut_sdp(shrunk_graph)  # Solve SDP for the updated graph
            correlations = calculate_sdp_correlations(B)  # Recompute correlations
            steps_since_recalculation = 0  # Reset the counter

        # Visualization after the shrinking step
       # plot_graph_step(shrunk_graph, f"Step {step}: After Shrinking (Merged {i} and {j})", partition)

    return shrunk_graph, partition, shrinking_steps


def generate_node_map(original_graph, shrunk_graph):
    """
    Generate a mapping of original graph nodes to reduced graph indices.

    Parameters:
        original_graph (rustworkx.PyGraph): The original graph before shrinking.
        shrunk_graph (rustworkx.PyGraph): The reduced graph after shrinking.

    Returns:
        dict: A mapping of original node indices to reduced graph indices.
    """
    # Get the list of nodes in the reduced graph
    reduced_nodes = list(shrunk_graph.node_indexes())

    # Create a mapping where reduced graph nodes are assigned new consecutive indices
    node_map = {node: idx for idx, node in enumerate(reduced_nodes)}

    return node_map




## quantum approach

def get_weight_matrix(graph):
    """
    Generate the weight matrix for a rustworkx.PyGraph, ensuring node indices are contiguous.

    Parameters:
        graph (rustworkx.PyGraph): The input graph.

    Returns:
        np.array: The weight matrix of the graph.
    """
    # Map node indices to contiguous indices
    nodes = list(graph.node_indexes())
    node_map = {node: idx for idx, node in enumerate(nodes)}

    # Create an empty weight matrix
    num_nodes = len(nodes)
    weight_matrix = np.zeros((num_nodes, num_nodes))

    # Fill the weight matrix
    for u, v, weight in graph.weighted_edge_list():
        weight_matrix[node_map[u], node_map[v]] = weight
        weight_matrix[node_map[v], node_map[u]] = weight  # Symmetric for undirected graphs

    return weight_matrix, node_map



def map_solution_to_original_nodes(solution, node_map):
    """
    Map the solution of the shrunk graph back to the original graph's nodes.

    Parameters:
        solution (list): The solution array for the reduced graph.
        node_map (dict): A dictionary mapping original node indices to reduced graph indices.

    Returns:
        dict: A dictionary representing the solution for the original graph.
    """
    # Invert the node_map to map from reduced graph indices to original node indices
    inverted_map = {v: k for k, v in node_map.items()}
    
    # Map the solution back to the original nodes
    original_solution = {inverted_map[i]: solution[i] for i in range(len(solution))}
    
    return original_solution


def reconstruct_solution(shrinking_steps, shrunk_solution):
    """
    Reconstruct the solution for the original graph from the shrunk graph's solution.

    Parameters:
        shrinking_steps (list of tuples): Each tuple represents a merging step:
            (node_i, node_j, sigma_ij, merged_into).
        shrunk_solution (dict): Solution of the shrunk graph (e.g., {0: 0, 1: 1, 5: 0}).

    Returns:
        dict: Solution for the original graph.
    """
    # Start with the solution for the shrunk graph
    full_solution = shrunk_solution.copy()

    # Backtrack through the merging steps
    for node_i, node_j, sigma_ij, merged_into in reversed(shrinking_steps):
        if sigma_ij > 0:  # Same partition
            full_solution[node_i] = full_solution[node_j]
        else:  # Opposite partitions
            full_solution[node_i] = 1 - full_solution[node_j]

    return full_solution


def visualize_maxcut_solution_two(graph, partition_assignments):
    """
    Visualize the Max-Cut solution based on the final partition assignments.

    Parameters:
        graph (rustworkx.PyGraph): The original graph.
        partition_assignments (dict): Dictionary mapping node IDs to their partition (0 or 1).
        
    Returns:
        None: Displays the Max-Cut visualization.
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    # Convert rustworkx graph to networkx graph for visualization
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(graph.node_indexes())
    for u, v, weight in graph.weighted_edge_list():
        nx_graph.add_edge(u, v, weight=weight)

    # Separate nodes into partitions
    partition_0 = [node for node, part in partition_assignments.items() if part == 0]
    partition_1 = [node for node, part in partition_assignments.items() if part == 1]

    # Determine edges crossing the cut
    crossing_edges = []
    cut_value = 0  # Initialize total cut value
    for u, v, weight in graph.weighted_edge_list():
        if partition_assignments[u] != partition_assignments[v]:  # Edge crosses the cut
            crossing_edges.append((u, v))
            cut_value += weight  # Add the weight of the edge to the cut value

    # Plot the graph
    pos = nx.spring_layout(nx_graph, seed=42)  # Layout for consistent visualization
    plt.figure(figsize=(10, 8))

    # Draw nodes with different colors for the two partitions
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=partition_0, node_color="lightblue", label="Partition 0", node_size=600)
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=partition_1, node_color="orange", label="Partition 1", node_size=600)

    # Draw all edges
    nx.draw_networkx_edges(nx_graph, pos, edge_color="gray", alpha=0.5, width=1)

    # Highlight crossing edges
    nx.draw_networkx_edges(nx_graph, pos, edgelist=crossing_edges, edge_color="red", width=2, label="Crossing Edges")

    # Add edge labels
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in nx_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels)

    # Add labels to nodes
    nx.draw_networkx_labels(nx_graph, pos, font_size=12, font_color="black")

    # Title and legend
    plt.title("Max-Cut Solution Visualization")
    plt.legend(loc="best")
    plt.show()

    # Print the solution details
    print("Partition 0 (Blue):", partition_0)
    print("Partition 1 (Orange):", partition_1)
    print("Edges Crossing the Cut:", crossing_edges)
    print("Total Cut Value:", cut_value)
