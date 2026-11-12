from matplotlib import pyplot as plt
import numpy as np
import time
from sym_state import grid_size, NUM_THREADS, criteria, h, limit_iterations
from solvers.jacobi import calc_jacobi_relaxation_cpu_mt
from numba import set_num_threads
from math import pi, sin, cos, tan

set_num_threads(NUM_THREADS)

rand_gen = np.random.default_rng(int(time.time()))

charge_density = np.zeros((grid_size, grid_size))

def populate_charges():
    global charge_density

    for r in range(0, 8):
        for theta in np.linspace(0, 2*pi, int(2*pi / 0.1)):
            charge_density[int(grid_size/2 + r * sin(theta))][int(grid_size/2 + 100 + r * cos(theta))] = -4*pi * 0.01


    for r in range(0, 8):
        for theta in np.linspace(0, 2*pi, int(2*pi / 0.1)):
            charge_density[int(grid_size/2 + r * sin(theta))][int(grid_size/2 - 100 + r * cos(theta))] = 4*pi * 0.01

    for y in range(0, grid_size):
        for r in range(0, 10):
            charge_density[y][min(int(r+1/tan(pi/4) * y), grid_size - 1)] = -4*pi * sin(y/(2*pi)) * 0.02

    for y in range(0, grid_size):
        for r in range(0, 10):
            charge_density[y][min(int(r + 1 / tan(-pi / 4) * y), grid_size - 1)] = -4 * pi * sin(y / (2 * pi)) * 0.04

    for r in range(140, 142):
        for theta in np.linspace(0, 2*pi, int(2*pi / 0.01)):
            charge_density[int(grid_size/2 + r * sin(theta))][int(grid_size/2 + r * cos(theta))] = 4*pi * cos(theta) * 0.05

#Also initial state  np.zeros((grid_size, grid_size))
swap_1 = charge_density.copy() #rand_gen.random(
  #  (grid_size, grid_size),
#   dtype= np.float32
#)

swap_2 = np.zeros((grid_size, grid_size))

populate_charges()

if __name__ == "__main__":

    state = True
    d = 0.0

    start = time.monotonic_ns()

    i = 0

    while True:

        t_s = time.monotonic_ns()

        if state:
            d = calc_jacobi_relaxation_cpu_mt(swap_1, swap_2, charge_density, 0, grid_size)
        else:
            d = calc_jacobi_relaxation_cpu_mt(swap_2, swap_1, charge_density, 0, grid_size)

        t_e = time.monotonic_ns()

        print(f"Iteration {i}: ended in ({round((t_e - t_s) * (10**-6), 2)}) milliseconds with difference: {d}")

        state = not state
        i += 1

        if d <= criteria or i > limit_iterations:
            break

    end = time.monotonic_ns()

    print(f"Finished in {round((end - start) * (10**-9), 2)} seconds")


    mm_size = float(grid_size) * h

    size_indecies = np.linspace(-mm_size / 2, mm_size / 2, grid_size)

    X, Y = (
        np.meshgrid(
            size_indecies,
            size_indecies,
        )
    )

    fig= plt.figure()

    ax = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection='3d')


    g = None
    Z = None
    if state:
        Z = swap_2
    else:
        Z = swap_1

    g = np.gradient(Z, h)

    lengths = np.hypot(g[0], g[1])

    lengths_saved = np.where(lengths == 0, 1, lengths)
    U = g[0] / lengths_saved
    V = g[1] / lengths_saved


    #ax.quiver(X, Y, U, V, color="C0", angles='xy',
    #          scale_units='xy', scale=5, width=.015)
    ax.pcolormesh(X, Y, Z)
    ax2.pcolormesh(X, Y, charge_density)
    ax3.plot_surface(X, Y, Z)

    plt.show()