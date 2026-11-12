from numba import jit
import numpy as np
import sym_state

# x-1,x+1 y-1, y+1
def _get_neighbours_cyclic(x: int, y: int) -> tuple[tuple[int, int], tuple[int, int]]:
    grid_size = sym_state.grid_size

    x_neighbours: tuple[int, int]
    y_neighbours: tuple[int, int]

    if x == 0:
        x_neighbours = (grid_size - 1, x + 1)
    elif x == grid_size - 1:
        x_neighbours = (x - 1, 0)
    else:
        x_neighbours = (x - 1, x + 1)

    if y == 0:
        y_neighbours = (grid_size - 1, y + 1)
    elif y == grid_size - 1:
        y_neighbours = (y - 1, 0)
    else:
        y_neighbours = (y - 1, y + 1)

    return x_neighbours, y_neighbours

get_neighbours_cyclic_cpu = jit(nopython=True, nogil=True, cache=True)(_get_neighbours_cyclic)

#Matrix expected to be with dimensions grid_size*grid_size
def _make_calc_cell_mean(
    get_members_func,
):
    def r (
        matrix,
        x,
        y
    ):
        n = get_members_func(x, y)

        left = matrix[y][n[0][0]]
        right = matrix[y][n[0][1]]
        top = matrix[n[1][0]][x]
        bottom = matrix[n[1][1]][x]

        m = np.mean(np.array([left, right, top, bottom]))

        return m, matrix[y][x] - m
    return r

calc_cell_mean_cpu = jit(nopython=True, nogil=True)(_make_calc_cell_mean(get_neighbours_cyclic_cpu))