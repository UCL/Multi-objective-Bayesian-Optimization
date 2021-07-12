import unittest
from src.MOBO_reg import MOBO, reformulation
from botorch.utils.sampling import draw_sobol_samples
import torch

class Testreformulation(unittest.TestCase):
    def test_input_linear_Transformation(self):
        #Test the input transformation is correct
        from test_functions.oka2 import problem, bounds
        ref = reformulation(problem, bounds(), minimize=False)
        standard_bounds = torch.zeros(2, 3)
        standard_bounds[1] = 1
        train_x = draw_sobol_samples(bounds=torch.Tensor(bounds()), n=1, q=10, seed=torch.randint(1000000, (1,)).item()).squeeze(
            0)
        train_obj = problem(train_x[0])


        train_x_reformulated = (train_x[[0]]- bounds()[0])/(bounds()[1]-bounds()[0])
        train_obj_ref = ref.evaluate(train_x_reformulated)
        self.assertAlmostEqual(train_obj_ref[0,0],train_obj[0])
        self.assertAlmostEqual(train_obj_ref[0,1],train_obj[1])
