import itertools
import numbers
import numpy as np
import stim

# input checks for functions below
def check_integer(name, value, minimum):
    if not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")

# "not minimum <= value <= maximum" also rejects NaN, since every comparison with NaN is false
def check_real(name, value, minimum, maximum):
    if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a number")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")

# CSS codes are a family of codes
# The codes we are interested in (toric & rotated surface) are members of this family - if you want to extend this project, start from here
# There are going to be common setup functions for both, so we will create them here
class CSSCode:

    # toric codes are essentially "doughnuts" so they don't have a boundary
    # rotated surface codes can be thought of as a "square"
    # we need some way of distinguishing between them
    has_boundary = False

    # in both codes, we know which stabilizers watch which qubits, but we need to make this easy to do computation over
    def build_stabilizer_matrix(self):
        self.stabilizer_matrix = np.zeros((len(self.stabilizer_qubits), self.number_of_qubits), dtype=np.uint8)
        for i, qubits in enumerate(self.stabilizer_qubits):
            self.stabilizer_matrix[i, qubits] = 1

        self.qubit_to_stabilizer = [[[] for _ in range(self.number_of_qubits)], [[] for _ in range(self.number_of_qubits)]]
        for i, qubits in enumerate(self.stabilizer_qubits):
            for qubit in qubits:
                self.qubit_to_stabilizer[self.stabilizer_type[i]][qubit].append(i)

    # we also need to determine which stabilizers are neighboring each other
    def build_adjacency_matrix(self):
        self.number_of_stabilizers = len(self.stabilizer_qubits)
        self.number_of_logical_errors = self.logical_errors.shape[1]
    
        self.number_of_nodes = self.number_of_stabilizers + (1 if self.has_boundary else 0)
        self.adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=bool)

        # using the stabilizer matrix, we can see if two stabilizers are neighbors
        neighbors = (self.stabilizer_matrix.astype(np.int64) @ self.stabilizer_matrix.T.astype(np.int64)) > 0
        np.fill_diagonal(neighbors, False)
        self.adjacency_matrix[:self.number_of_stabilizers, :self.number_of_stabilizers] = neighbors

        # on the rotated surface code, to make error strings always trigger 2 endpoints, we can create an "imaginary" stabilizer that neighbors every other boundary stabilizer, otherwise only 1 might be triggered
        if self.has_boundary:
            self.adjacency_matrix[self.number_of_stabilizers, :self.number_of_stabilizers] = self.adjacency_matrix[:self.number_of_stabilizers, self.number_of_stabilizers] = self.boundary_adjacent_stabilizers

    # build program that Stim can use
    def build_circuit(self):
        lines = []
        for stabilizer, qubits in enumerate(self.stabilizer_qubits):
            pauli = "X" if self.stabilizer_type[stabilizer] == 0 else "Z"
            lines.append("MPP " + "*".join(f"{pauli}{qubit}" for qubit in qubits))
        self.circuit = stim.Circuit("\n".join(lines))

    def build_structures(self):
        self.build_stabilizer_matrix()
        self.build_adjacency_matrix()
        self.build_circuit()

    # all these codes need some way of simulating different types of error models
    NOISE_MODELS = (
        "depolarizing", # X error, Z error, Y error (both X and Z together) equally likely
        "biased", # Z errors far more likely
        "x_only",
        "y_only",
        "pauli" # general X/Y/Z channel, set by probability_x_error, probability_y_error and probability_z_error
    )

    def add_noise_to_circuit(self, probability_error, noise_model="depolarizing", bias=10.0, probability_x_error = 1 / 3, probability_y_error = 1 / 3, probability_z_error = 1 / 3):
        if noise_model not in self.NOISE_MODELS:
            raise ValueError(f"noise_model must be one of {self.NOISE_MODELS}")
        check_real("probability_error", probability_error, 0.0, 1.0)
        check_real("bias", bias, 0.0, np.finfo(float).max)
        for name, value in (("probability_x_error", probability_x_error), ("probability_y_error", probability_y_error), ("probability_z_error", probability_z_error)):
            check_real(name, value, 0.0, np.finfo(float).max)
        if probability_x_error + probability_y_error + probability_z_error == 0:
            raise ValueError("probability_x_error, probability_y_error and probability_z_error cannot all be 0")

        circuit = stim.Circuit()
        if noise_model == "depolarizing":
            circuit.append("DEPOLARIZE1", range(self.number_of_qubits), probability_error)
        elif noise_model == "biased":
            probability_x = probability_y = probability_error / (2.0 * (1.0 + bias))
            probability_z = probability_error * bias / (1.0 + bias)
            circuit.append("PAULI_CHANNEL_1", range(self.number_of_qubits), [probability_x, probability_y, probability_z])
        elif noise_model == "x_only":
            circuit.append("X_ERROR", range(self.number_of_qubits), probability_error)
        elif noise_model == "y_only":
            circuit.append("Y_ERROR", range(self.number_of_qubits), probability_error)
        else:
            total = probability_x_error + probability_y_error + probability_z_error
            circuit.append("PAULI_CHANNEL_1", range(self.number_of_qubits), [probability_error * probability_x_error / total, probability_error * probability_y_error / total, probability_error * probability_z_error / total])
        return circuit

    # now let's simulate these errors
    def simulate_errors_in_circuit(self, number_of_runs, probability_error, noise_model="depolarizing", seed=None, bias=10.0, probability_x_error = 1 / 3, probability_y_error = 1 / 3, probability_z_error = 1 / 3):
        check_integer("number_of_runs", number_of_runs, 1)
        if seed is not None:
            check_integer("seed", seed, 0)
            if seed >= 2**64:
                raise ValueError("seed must be less than 2**64")

        circuit = self.add_noise_to_circuit(probability_error, noise_model, bias, probability_x_error, probability_y_error, probability_z_error)
        
        # need to simulate errors then add this to the specific code
        simulator = stim.FlipSimulator(batch_size=number_of_runs, num_qubits=self.number_of_qubits, disable_stabilizer_randomization=True, seed=seed)
        simulator.do(circuit)
        errors = simulator.peek_pauli_flips()
        simulator.do(self.circuit)

        # now we need to return the errors
        syndrome = simulator.get_measurement_flips().T.astype(np.uint8)
        x_errors = np.zeros((number_of_runs, self.number_of_qubits), np.uint8)
        z_errors = np.zeros((number_of_runs, self.number_of_qubits), np.uint8)
        for i, error in enumerate(errors):
            x_error, z_error = error.to_numpy()
            x_errors[i], z_errors[i] = x_error, z_error
        
        return x_errors, z_errors, syndrome

# In the toric code, blocks on a grid are wrapped into a doughnut (hence toric)
# Every stabilizer watches 4 blocks
class ToricCode(CSSCode):
    
    """
    On an LxL periodic lattice:
        - there are 2(L^2) edges -> these correspond to the physical qubits
        - there are L^2 intersections and L^2 squares -> these correspond to the 2(L^2) X and Z stabilizers
    """

    def __init__(self, lattice_size):
        if not isinstance(lattice_size, numbers.Integral) or isinstance(lattice_size, (bool, np.bool_)):
            raise TypeError("lattice_size must be an integer")
        lattice_size = int(lattice_size)
        if lattice_size <= 1:
            raise ValueError("lattice_size must be an integer greater than 1")

        self.distance = lattice_size
        self.number_of_qubits = 2 * lattice_size * lattice_size

        # now we need a way of navigating to a specific qubit
        # I am going to split by horizontal and vertical edges - that is the standard way to do this
        horizontal_edge_index = lambda x, y: (y % lattice_size) * lattice_size + (x % lattice_size)
        vertical_edge_index = lambda x, y: lattice_size * lattice_size + horizontal_edge_index(x, y)

        # now we need to set up the stabilizers
        self.stabilizer_qubits = []
        self.stabilizer_type = []
        self.stabilizer_coordinates = []

        # x stabilizer = 0, z stabilizer = 1
        for stabilizer_type in (0, 1):
            for y, x in itertools.product(range(lattice_size), repeat=2):
                self.stabilizer_qubits.append([horizontal_edge_index(x, y), horizontal_edge_index(x - 1, y), vertical_edge_index(x, y), vertical_edge_index(x, y - 1)] if stabilizer_type == 0 else [horizontal_edge_index(x, y), horizontal_edge_index(x, y + 1), vertical_edge_index(x, y), vertical_edge_index(x + 1, y)])
                self.stabilizer_type.append(stabilizer_type)
                self.stabilizer_coordinates.append((x, y))
        self.stabilizer_type = np.asarray(self.stabilizer_type, dtype=np.int64)
        self.stabilizer_coordinates = np.array(self.stabilizer_coordinates, float)

        # euclidean distance to find distance from center
        self.position = np.sqrt(((self.stabilizer_coordinates - lattice_size / 2) ** 2).sum(1, keepdims=True))

        # there are some types of errors that can go undetected but cause a logical error
        # in toric codes, those errors are ones that are closed horizontal and vertical loops of errors
        self.logical_errors = np.zeros((2, 2, self.number_of_qubits), np.uint8)
        for i in range(lattice_size):
            self.logical_errors[0, 0, vertical_edge_index(i, 0)] = 1
            self.logical_errors[1, 0, horizontal_edge_index(i, 0)] = 1
            self.logical_errors[0, 1, horizontal_edge_index(0, i)] = 1
            self.logical_errors[1, 1, vertical_edge_index(0, i)] = 1

        # finally build shared structures
        self.build_structures()

# In the rotated surface code, blocks are mapped into a square grid
# Stabilizers in the middle watch 4 blocks, and ones on the edges watch 2 blocks
class RotatedSurfaceCode(CSSCode):

    """
    On an LxL lattice:
        - there are L^2 intersections -> these correspond to the physical qubits
        - there are (L-1)^2 squares and 2(L-1) boundary edges -> these correspond to the locations of the (L^2)-1 X and Z alternating stabilizers
    """

    has_boundary = True

    def __init__(self, lattice_size):
        if not isinstance(lattice_size, numbers.Integral) or isinstance(lattice_size, (bool, np.bool_)):
            raise TypeError("lattice_size must be an integer")
        lattice_size = int(lattice_size)
        if lattice_size <= 1:
            raise ValueError("lattice_size must be an integer greater than 1")
            
        self.distance = lattice_size
        self.number_of_qubits = lattice_size * lattice_size

        # now we need a way of navigating to a specific qubit
        qubit_index = lambda row, column: row * lattice_size + column

        # now we need to set up the stabilizers
        self.stabilizer_qubits = []
        self.stabilizer_type = []
        self.stabilizer_coordinates = []

        for i, j in itertools.product(range(-1, lattice_size), repeat=2):
            stabilizer_type = 0 if (i + j) % 2 == 0 else 1 # x stabilizer = 0, z stabilizer = 1
            qubits = [qubit_index(row, column) for row in (i, i + 1) for column in (j, j + 1) if 0 <= row < lattice_size and 0 <= column < lattice_size]

            if len(qubits) == 4 or (len(qubits) == 2 and ((stabilizer_type == 0 and i in (-1, lattice_size - 1)) or (stabilizer_type == 1 and j in (-1, lattice_size - 1)))):
                self.stabilizer_qubits.append(qubits)
                self.stabilizer_type.append(stabilizer_type)
                self.stabilizer_coordinates.append([i + 0.5, j + 0.5])
        self.stabilizer_type = np.asarray(self.stabilizer_type, dtype=np.int64)
        self.stabilizer_coordinates = np.array(self.stabilizer_coordinates, float)

        # calculate spatial center of mass
        self.position = np.array([[np.mean([qubit // lattice_size for qubit in qubits]), np.mean([qubit % lattice_size for qubit in qubits])] for qubits in self.stabilizer_qubits])

        # there are some types of errors that can go undetected but cause a logical error
        # in rotated surface codes, those errors are ones that go from boundary to boundary
        self.logical_errors = np.zeros((2, 1, self.number_of_qubits), np.uint8)
        self.logical_errors[0, 0, [qubit_index(row, 0) for row in range(lattice_size)]] = 1
        self.logical_errors[1, 0, [qubit_index(0, column) for column in range(lattice_size)]] = 1

        # stabilizers are on the boundary iff one of its qubits are not watched by any other same-type stabilizer
        watched_qubits = np.zeros((2, self.number_of_qubits), np.int64)
        for stabilizer_type, qubits in zip(self.stabilizer_type, self.stabilizer_qubits):
            watched_qubits[stabilizer_type, qubits] += 1
        self.boundary_adjacent_stabilizers = np.array([bool((watched_qubits[stabilizer_type, qubits] == 1).any()) for stabilizer_type, qubits in zip(self.stabilizer_type, self.stabilizer_qubits)])

        # finally build shared structures
        self.build_structures()