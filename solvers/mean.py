import sym_state
from numba import jit, prange
from utility import calc_cell_mean_cpu

def _make_calc_mean(
        mean_func,
        range_function
):
    def r(
        inp,
        result,
        from_row: int,
        end_row: int
    ):
        max_difference = 0.0
        for y in range_function(from_row, end_row):
            for x in range(0, sym_state.grid_size):
                m = mean_func(inp, x, y)
                result[y][x] = m[0]
                max_difference = max(max_difference, m[1])

        return max_difference

    return r

calc_mean_cpu = jit(nopython=True, nogil=True)(_make_calc_mean(calc_cell_mean_cpu, range))
calc_mean_cpu_mt = jit(nopython=True, nogil=True, parallel=True)(
    _make_calc_mean(calc_cell_mean_cpu, prange)
)
