
import sym_state
from utility import calc_cell_mean_cpu
from numba import jit,prange

def _make_calc_jacobi_relaxation(
        mean_func,
        range_function
):
    def r(
        inp,
        result,
        laplacian_value,
        from_row: int,
        end_row: int
    ):
        max_difference = 0.0
        for y in range_function(from_row, end_row):
            for x in range(0, sym_state.grid_size):
                m = mean_func(inp, x, y)
                laplacian_part = ((sym_state.h ** 2) * laplacian_value[y][x]) / 4.0
                result[y][x] = m[0] + laplacian_part
                max_difference = max(max_difference, abs(result[y][x] - inp[y][x]))
        return max_difference

    return r


calc_jacobi_relaxation_cpu = jit(nopython=True, nogil=True)(_make_calc_jacobi_relaxation(calc_cell_mean_cpu, range))
calc_jacobi_relaxation_cpu_mt = jit(nopython=True, nogil=True, parallel=True)(
    _make_calc_jacobi_relaxation(calc_cell_mean_cpu, prange)
)
