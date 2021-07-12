import os
import torch

import numpy as np

np.random.seed(0)
import random

random.seed(0)
torch.manual_seed(0)

from botorch.test_functions.multi_objective import C2DTLZ2
from MOBO_reg import MOBO


d = 12
M = 2
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
from botorch.test_functions.multi_objective import BraninCurrin


# problem = BraninCurrin(negate=True).to(**tkwargs)


def problem(x):

    transl = 1/ np.sqrt(2)
    part1 = (x[0] - transl)**2 + (x[1] - transl)**2
    part2 = (x[0] + transl)**2 + (x[1] + transl)**2
    f1 = 1 - np.exp(-part1)
    f2 = 1 - np.exp(-part2)
    return [f1, f2]

def oka2( x ):

    f1 = x[0]
    f2 = 1 - 1/(4*np.pi**2) * (x[0]+np.pi)**2 +\
           (abs(x[1] - 5*np.cos(x[0]) ))**(1/3.)\
         + (abs( x[2]- 5*np.sin(x[0]) ))**(1/3.)
    return [f1, f2]

bounds  = np.array([[-np.pi, -5.,-5.],[np.pi,5.,5.]])
MOBO_TRY = MOBO(oka2, bounds,minimize=True, N_iteration=40)
hvs_qehvi_all = MOBO_TRY.perform_MOBO()

print(2)