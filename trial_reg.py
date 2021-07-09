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


problem = BraninCurrin(negate=True).to(**tkwargs)


def problem(x):


    return [x[0]**2, np.sin(0.1*x[1])]



bounds  = np.array([[0., 0.],[1.,1.]])
MOBO_TRY = MOBO(problem, bounds)
hvs_qehvi_all = MOBO_TRY.perform_MOBO()

print(2)