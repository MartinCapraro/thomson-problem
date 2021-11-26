import sys
import argparse
import logging

import numpy as np
from scipy.spatial import distance_matrix

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

# Logging options
logging.basicConfig(
    # filename=os.path.join(dir_path, 'thomson_problem.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# Define the parser used to process the command line input and then add a bunch of arguments
parser = argparse.ArgumentParser(
            prog='thomson',
            description='Find approximate solutions to the Thomson problem in arbitrary dimension',
            allow_abbrev=False
        )

parser.add_argument(
            '-n',
            metavar='dimension',
            type=int,
            required=False,
            default=3,
            help='the dimension of Euclidean space (default: 3)'
        )

parser.add_argument(
            '-m',
            metavar='points',
            type=int,
            required=False,
            default=8,
            help='the number of point particles on the (n-1) sphere (default: 8)'
        )

parser.add_argument(
            '-p',
            metavar='power',
            type=int,
            required=False,
            default=1,
            help='the power of the inverse radius in the potential energy function (default: 1)'
        )


parser.add_argument(
            '-eps',
            metavar='epsilon',
            type=float,
            required=False,
            default=0.1,
            help='the gradient descent epsilon - the step size (default: 0.1)'
        )


parser.add_argument(
            '-max_iter',
            metavar='max_iteration',
            type=int,
            required=False,
            default=1e3,
            help='the number of steps of gradient descent the program will take (default: 1000)'
        )


def generate_random_vectors_of_unit_norm(n, m, fixed_seed=False):
    if fixed_seed:
        np.random.seed(0)

    # List comprehensions are great
    vector_lst = [generate_random_vector_of_unit_norm(n) for _ in range(m)]

    return np.stack(vector_lst, axis=0)


def generate_random_vector_of_unit_norm(n):

    x = np.random.standard_normal(n)
    x = x/np.linalg.norm(x)

    return x

def calculate_total_energy(array, p=1):
    n = len(array[0])
    m = len(array)

    energy = 0
    dm = distance_matrix(array, array)

    mask = np.ones(dm.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    energy = (1.0/dm[mask]).sum()/2.0

    # This is equivalent to the calculation above, not
    # sure which is faster or whether it matters
    # for i in range(m):
    #     for j in range(i+1,m):
    #         energy += 1.0/dm[i, j]

    return energy


def grad_func(array, p=1):
    m = len(array)
    n = len(array[0])

    grad_mtx = np.zeros([m, n])
    dm = distance_matrix(array, array)

    for i in range(m):
        grad = 0.0
        # iterate over all points j where i != j
        for j in range(m):
            if i == j:
                pass
            else:
                new_vec = (array[i] - array[j])/dm[i, j]**(p+1)
                grad = grad + new_vec

        grad_mtx[i] = -2.0*p*grad

    return grad_mtx


def minimise_energy(w_init, max_iter=1e3, eps=0.01, p=1):
    logging.debug("Starting energy minimisation")
    v = w_init
    e = calculate_total_energy(w_init, p=p)
    logging.debug("Energy of inital configuration is: {}".format(e))

    # iterate over max_energy_iter
    for n in range(int(max_iter)):
        logging.debug("{}/{}".format(n, int(max_iter)))
        # One step of vanilla gradient descent
        v = v - eps*grad_func(v, p=p)

        # Normalise each row so that the points are still on the unit sphere
        norm_of_rows = np.linalg.norm(v, axis=1)
        v = v/norm_of_rows[:, np.newaxis]

        energy = calculate_total_energy(v)
        logging.debug("Energy : {}".format(energy))

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

    np.savetxt('n={}_m={}_p={}.txt'.format(n, m, p), configuration)

    logging.info("Minimised energy: {}".format(energy))

    logging.debug("Final configuration after energy minimisation:")
    logging.debug(repr(configuration))


def main():
    args = parser.parse_args()

    n = args.n
    m = args.m
    p = args.p
    eps = args.eps
    max_iter = args.max_iter

    run_once(n=n, m=m, p=p, eps=eps, max_iter=max_iter)

if __name__ == '__main__':
    main()

# TODO: add a boolean flag to write the position of the particles to a file
