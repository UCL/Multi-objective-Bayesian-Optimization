import numpy as np


def problem(x):

    transl = 1/ np.sqrt(2)
    part1 = (x[0] - transl)**2 + (x[1] - transl)**2
    part2 = (x[0] + transl)**2 + (x[1] + transl)**2
    f1 = 1 - np.exp(-part1)
    f2 = 1 - np.exp(-part2)
    return [f1, f2]

def bounds():

    return np.array([[-np.pi, -5.,-5.],[np.pi,5.,5.]])