import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

# Logging options
logging.basicConfig(
    # filename=os.path.join(dir_path, 'thomson_problem.log'),
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Define the parser used to process the command line input and then add a bunch of arguments
parser = argparse.ArgumentParser(
    prog="python thomson.py",
    description="Find approximate solutions to the Thomson problem in arbitrary dimension",
    allow_abbrev=False,
)

parser.add_argument(
    "-n",
    metavar="dimension",
    type=int,
    required=False,
    default=3,
    help="the dimension of Euclidean space (default: 3)",
)

parser.add_argument(
    "-m",
    metavar="points",
    type=int,
    required=False,
    default=8,
    help="the number of point particles on the (n-1) sphere (default: 8)",
)

parser.add_argument(
    "-p",
    metavar="power",
    type=int,
    required=False,
    default=1,
    help="the power of the inverse radius in the potential energy function (default: 1)",
)


parser.add_argument(
    "-eps",
    metavar="epsilon",
    type=float,
    required=False,
    default=0.1,
    help="the gradient descent epsilon - i.e. the step size (default: 0.1)",
)


parser.add_argument(
    "-max_iter",
    metavar="max_iteration",
    type=int,
    required=False,
    default=1e3,
    help="the number of steps of gradient descent the program will take (default: 1000)",
)


parser.add_argument(
    "--plot",
    metavar="plot",
    required=False,
    default=False,
    action=argparse.BooleanOptionalAction,
    help="plot the found solution - only valid when n = 3 (default: false)",
)


def generate_random_vectors_of_unit_norm(n, m, fixed_seed=False):
    if fixed_seed:
        np.random.seed(0)

    # List comprehensions are great
    vector_lst = [generate_random_vector_of_unit_norm(n) for _ in range(m)]

    return np.stack(vector_lst, axis=0)


def generate_random_vector_of_unit_norm(n):
    x = np.random.standard_normal(n)
    x = x / np.linalg.norm(x)

    return x


def calculate_total_energy(array, p=1):
    energy = 0

    # Compute the pairwise differences
    diff = array[:, None, :] - array[None, :, :]
    dm = np.sum(diff**2, axis=-1) ** 0.5

    mask = np.ones(dm.shape, dtype=bool)
    np.fill_diagonal(mask, 0)

    # The energy for the potential V(r) = r**(-p) is sum (dm[mask]^-p) / 2.0
    energy = (dm[mask] ** (-p)).sum() / 2.0

    return energy


# I'm leaving this here as a reminder of how I used to calculate this
def grad_func_old(array, p=1):
    # Compute the pairwise differences
    diff = array[:, None, :] - array[None, :, :]

    # Compute the square of the distance matrix and add the unit matrix to
    # avoid division by zero when calculating the inverse distances
    dms = np.sum(diff**2, axis=-1) + np.eye(len(array))

    # Inverse of the distance matrix squared
    inv_distances = dms ** (-(p + 1) / 2)

    # Matrix of gradients
    grad_mtx = -2.0 * p * np.sum(diff * inv_distances[..., None], axis=1)

    return grad_mtx


def grad_func(array, p=1):
    # Compute pairwise squared differences
    sq_diff = (
        np.sum(array[:, None, :] ** 2, axis=-1)
        + np.sum(array[None, :, :] ** 2, axis=-1)
        - 2.0 * np.dot(array, array.T)
    )

    # Add the unit matrix to avoid division by zero when calculating the
    # inverse distance
    sq_diff = sq_diff + np.eye(len(array))

    # Inverse of the distance matrix squared
    inv_distances = sq_diff ** (-(p + 1) / 2)

    # Pairwise differences
    diff = array[:, None, :] - array[None, :, :]

    # Matrix of gradients
    grad_mtx = -2.0 * p * np.einsum("ijk,ij->ik", diff, inv_distances)

    return grad_mtx


# https://en.wikipedia.org/wiki/Barzilai%E2%80%93Borwein_method
def barzilai_borwein(v, v_prev, g, g_prev, eps):
    s = v - v_prev  # Displacement
    y = g - g_prev  # Change in gradient

    s_dot_y = np.sum(s * y)

    if s_dot_y > 0:
        eps = np.sum(s * s) / s_dot_y
        return eps
    else:
        return eps


def minimise_energy(w_init, max_iter=1e3, eps=0.01, p=1):
    logging.debug("Starting energy minimisation")
    v = w_init

    # Pre-calculate the gradient for the first step
    g = grad_func(v, p=p)

    e = calculate_total_energy(w_init, p=p)
    logging.debug("Energy of initial configuration is: {}".format(e))

    # iterate over max_iter steps
    for n in range(int(max_iter)):
        logging.debug("{}/{}".format(n, int(max_iter)))

        # Store the previous position and gradient
        v_prev = v
        g_prev = g

        # Take one step of vanilla gradient descent using the current eps
        v = v - eps * grad_func(v, p=p)

        # Normalise each row so that the points are still on the unit sphere
        norm_of_rows = np.linalg.norm(v, axis=1)
        v = v / norm_of_rows[:, np.newaxis]

        # Calculate the new gradient at the new position
        g = grad_func(v, p=p)

        eps = barzilai_borwein(v, v_prev, g, g_prev, eps)
        logging.debug(f"Eps: {eps}")

        energy = calculate_total_energy(v, p=p)
        logging.debug(f"Energy: {energy}")

    return v


def run_once(n, m, p, eps, max_iter):
    w_init = generate_random_vectors_of_unit_norm(n=n, m=m)
    logging.info("Initial energy: {}".format(calculate_total_energy(w_init, p=p)))
    logging.debug("Initial configuration:")
    logging.debug("{}".format(repr(w_init)))

    # sanity check that the norms are close to unity
    logging.debug("Norms of init vectors are:")
    [logging.debug(np.linalg.norm(w_init[i])) for i in range(m)]

    configuration = minimise_energy(w_init=w_init, eps=eps, max_iter=max_iter, p=p)
    energy = calculate_total_energy(configuration, p=p)

    np.savetxt(output_file_path(n, m, p), configuration, delimiter=",")

    logging.info("Minimised energy: {}".format(energy))

    logging.debug("Final configuration after energy minimisation:")
    logging.debug(repr(configuration))


def plot_points_on_two_sphere(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Generate spherical data
    r = 1
    theta = np.linspace(0, 2.0 * np.pi, 40)
    phi = np.linspace(0, np.pi, 40)

    # Convert to cartesian coordinates
    x = r * np.outer(np.cos(theta), np.sin(phi))
    y = r * np.outer(np.sin(theta), np.sin(phi))
    z = r * np.outer(np.ones(np.size(theta)), np.cos(phi))

    ax.plot_surface(x, y, z, rstride=1, cstride=1, color="c", alpha=0.3, linewidth=1)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="r")

    ax.set_xlim(left=-1, right=1)
    ax.set_ylim(bottom=-1, top=1)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def output_file_path(n, m, p):
    # TODO: return absolute instead of relative path
    return f"./output/n={n}_m={m}_p={p}.csv"


def main():
    args = parser.parse_args()

    n = args.n
    m = args.m
    p = args.p
    eps = args.eps
    max_iter = args.max_iter
    plot = args.plot

    run_once(n=n, m=m, p=p, eps=eps, max_iter=max_iter)

    if plot is True and n == 3:
        points = np.genfromtxt(output_file_path(n, m, p), delimiter=",")
        plot_points_on_two_sphere(points)
    elif plot and n != 3:
        logging.error("Can only plot solutions for n=3 (i.e. the 2-sphere)")


if __name__ == "__main__":
    main()
