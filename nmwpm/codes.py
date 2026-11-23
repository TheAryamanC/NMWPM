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

# In the rotated surface code, blocks are mapped into a square grid
# Stabilizers in the middle watch 4 blocks, and ones on the edges watch 2 blocks
class RotatedSurfaceCode(CSSCode):

    has_boundary = True

    """
    On an LxL periodic lattice:
        - there are L^2 intersections -> these correspond to the physical qubits
        - there are (L-1)^2 squares and 2(L-1) boundary edges -> these correspond to the (L^2)-1 X and Z stabilizers
    """
    def __init__(self, lattice_size):
        self.distance = lattice_size
        self.number_of_qubits = lattice_size * lattice_size

        # now we need a way of navigating to a specific qubit
        qubit_index = lambda row, col: row * lattice_size + col