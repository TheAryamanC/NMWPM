# CSS codes are a family of codes
# The codes we are interested in (toric & rotated surface) are members of this family - if you want to extend this project, start from here
# There are going to be common setup functions for both, so we will create them here
class CSSCode:
    pass

# In the toric code, blocks on a grid are wrapper into a doughnut (hence toric)
# Every stabilizer watches 4 blocks
class ToricCode(CSSCode):
    pass

# In the rotated surface code, blocks are mapped into a square grid
# Stabilizers in the middle watch 4 blocks, and ones on the edges watch 2 blocks
class RotatedSurfaceCode(CSSCode):
    pass