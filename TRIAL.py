import os
import torch

import numpy as np

np.random.seed(0)
import random

random.seed(0)
torch.manual_seed(0)

from botorch.test_functions.multi_objective import C2DTLZ2
from MOBO import MOBO


d = 12
M = 2
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
problem = C2DTLZ2(dim=d, num_objectives=M, negate=True).to(**tkwargs)


MOBO_TRY = MOBO(problem)
hvs_qehvi_all = MOBO_TRY.perform_MOBO()

print(2)