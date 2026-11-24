import itertools
import numpy as np

# CSS codes are a family of codes
# The codes we are interested in (toric & rotated surface) are members of this family - if you want to extend this project, start from here
# There are going to be common setup functions for both, so we will create them here
class CSSCode:

    # toric codes are essentially "doughnuts" so they don't have a boundary
    # rotated surface codes can be thought of as a "square"
    # we need some way of distinguishing between them
    has_boundary = False

# In the toric code, blocks on a grid are wrapper into a doughnut (hence toric)
# Every stabilizer watches 4 blocks
class ToricCode(CSSCode):
    
    """
    On an LxL periodic lattice:
        - there are 2(L^2) edges -> these correspond to the physical qubits
        - there are L^2 intersections and L^2 squares -> these correspond to the 2(L^2) X and Z stabilizers
    """
    def __init__(self, lattice_size):
        self.distance = lattice_size
        self.number_of_qubits = 2 * lattice_size * lattice_size

        # now we need a way of navigating to a specific qubit
        # I am going to split by horizontal and vertical edges - that is the standard way to do this
        horizontal_edge_index = lambda x, y: (y % lattice_size) * L + (x % lattice_size)
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
        self.stabilizer_coordinates = np.array(self.stabilizer_coordinates, float)

        # euclidean distance to find distance from center
        self.position = np.sqrt(((self.stabilizer_coordinates - lattice_size / 2) ** 2).sum(1, keepdims=True))

        # there are some types of errors that can go undetected but cause a logical error
        # in toric codes, those errors are ones that are closed horizontal and vertical loops of errors
        self.logical_errors = np.zeros((2, 2, self.number_of_qubits), np.uint8)
        for i in range(L):
            self.logical_errors[0, 0, vertical_edge_index(i, 0)] = 1
            self.logical_errors[1, 0, horizontal_edge_index(i, 0)] = 1
            self.logical_errors[0, 1, horizontal_edge_index(0, i)] = 1
            self.logical_errors[1, 1, vertical_edge_index(0, i)] = 1

# In the rotated surface code, blocks are mapped into a square grid
# Stabilizers in the middle watch 4 blocks, and ones on the edges watch 2 blocks
class RotatedSurfaceCode(CSSCode):

    has_boundary = True

    """
    On an LxL periodic lattice:
        - there are L^2 intersections -> these correspond to the physical qubits
        - there are (L-1)^2 squares and 2(L-1) boundary edges -> these correspond to the locations of the (L^2)-1 X and Z alternating stabilizers
    """
    def __init__(self, lattice_size):
        self.distance = lattice_size
        self.number_of_qubits = lattice_size * lattice_size

        # now we need a way of navigating to a specific qubit
        qubit_index = lambda row, column: row * lattice_size + column

        # now we need to set up the stabilizers
        self.stabilizer_qubits = []
        self.stabilizer_type = []
        self.stabilizer_coordinates = []

        for i, j in itertools.product(range(-1, L), repeat=2):
            stabilizer_type = 0 if (i + j) % 2 == 0 else 1 # x stabilizer = 0, z stabilizer = 1
            qubits = [qubit_index(row, column) for row in (i, i + 1) for column in (j, j + 1) if 0 <= row < lattice_size and 0 <= column < lattice_size]

            if len(qubits) == 4 or (len(qubits) == 2 and ((stabilizer_type == 0 and i in (-1, lattice_size - 1)) or (stabilizer_type == 1 and i in (-1, lattice_size)))):
                self.stabilizer_qubits.append(qubits)
                self.stabilizer_type.append(stabilizer_type)
                self.stabilizer_coordinates.append([i + 0.5, j + 0.5])
        self.stabilizer_coordinates = np.array(self.stabilizer_coordinates, float)

        # calculate spatial center of mass
        self.position = np.array([[np.mean([qubit // lattice_size for qubit in qubits]), np.mean([qubit % lattice_size for qubit in qubits])] for qubits in self.stabilizer_qubits])

        # there are some types of errors that can go undetected but cause a logical error
        # in rotated surface codes, those errors are ones that go from boundary to boundary
        self.logical_errors = np.zeros((2, 1, self.number_of_qubits), np.uint8)
        self.logical_errors[0, 0, [qubit_index(row, 0) for row in range(lattice_size)]] = 1
        self.logical_errors[0, 0, [qubit_index(0, column) for column in range(lattice_size)]] = 1