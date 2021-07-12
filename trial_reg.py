import torch

import numpy as np

np.random.seed(0)
import random

random.seed(0)
torch.manual_seed(0)

from src.MOBO_reg import MOBO
from test_functions.oka2 import problem, bounds

d = 12
M = 2
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


# problem = BraninCurrin(negate=True).to(**tkwargs)





bounds  = bounds()#np.array([[-np.pi, -5.,-5.],[np.pi,5.,5.]])
MOBO_TRY = MOBO(problem, bounds,minimize=True, N_iteration=40)
hvs_qehvi_all = MOBO_TRY.perform_MOBO()

print(2)