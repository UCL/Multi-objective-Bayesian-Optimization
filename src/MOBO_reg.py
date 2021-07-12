import os
import torch

import numpy as np

np.random.seed(0)
import random

random.seed(0)
torch.manual_seed(0)

from botorch.test_functions.multi_objective import C2DTLZ2
import tqdm

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
import sys
from time import sleep


class reformulation:
    # This class aims to transform problems from numpy to torch and normalize inputs
    def __init__(self, problem, bounds, minimize, output_numpy=True):
        self. minimize =minimize
        if self.minimize:
            self.factor = -1.
        else:
            self.factor = 1.
        self.output_numpy = output_numpy
        self.bounds = torch.from_numpy(bounds).type(torch.FloatTensor)
        self.dim     = bounds.shape[1]
        # self.dim_y   = problem.shape[1]
        self.problem = problem

    def evaluate(self, x_torch_norm):
        self.dim_x = x_torch_norm.shape[1]
        #
        ub     = self.bounds[1,:]
        lb     = self.bounds[0,:]
        #here normalize
        x_torch = x_torch_norm * (ub-lb) + lb
        x       = x_torch.detach().numpy().reshape(-1, self.dim_x)


        for i in range(x.shape[0]):

            if self.output_numpy == False:
                f = (self.problem(x_torch[i, :])).type(torch.FloatTensor)
            else:
                f = torch.from_numpy(np.array(self.problem(x[i,:]))).type(torch.FloatTensor)

            if i ==0:
                f_store = torch.stack([self.factor*f])
            else:
                f_store =  torch.vstack([f_store,self.factor*f.reshape(1,-1)])
        return f_store







class MOBO:
    # This builds the MO
    # The user can input the problem bounds in numpy format.
    # BATCH_SIZE is integer (q) for the batch of the qEHVI
    # N_iteration is the number of iteration that are alg perform
    # If the optimization is a minimization let minimize = True
    def __init__(self, problem, bounds, BATCH_SIZE=3,N_iteration=25, minimize=False):
        reformulated_problem = reformulation(problem,bounds, minimize)
        self.N_BATCH = N_iteration
        self.problem = reformulated_problem

        self.tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
        self.SMOKE_TEST = os.environ.get("SMOKE_TEST")

        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_RESTARTS = 30 if not self.SMOKE_TEST else 2
        self.RAW_SAMPLES = 1024 if not self.SMOKE_TEST else 4
        self.standard_bounds = torch.zeros(2, self.problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1




    def generate_initial_data(self, n):
        # generate training data
        problem = self.problem
        train_x = draw_sobol_samples(bounds=self.standard_bounds, n=1, q=n, seed=torch.randint(1000000, (1,)).item()).squeeze(
            0)
        train_obj = problem.evaluate(train_x)
        # negative values imply feasibility in botorch

        self.ref_point = train_obj.min(0).values#problem.ref_point#problem.ref_point#feas_train_obj.max(0).values#

        return train_x, train_obj

    def initialize_model(self,train_x, train_obj):
        # define models for objective and constraint
        model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def optimize_qehvi_and_get_observation(self,model, train_obj, sampler):
        """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
        # partition non-dominated space into disjoint rectangles
        problem = self.problem
        partitioning = NondominatedPartitioning(ref_point=self.ref_point, Y=train_obj)
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point.tolist(),  # use known reference point
            partitioning=partitioning,
            sampler=sampler,
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.standard_bounds,
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
            sequential=True,
        )
        # observe new values
        new_x = unnormalize(candidates.detach(), bounds=self.standard_bounds)
        new_obj = problem.evaluate(new_x)
        return new_x, new_obj

    # def get_constrained_mc_objective(self, train_obj, train_con, scalarization):
    #     """Initialize a ConstrainedMCObjective for qParEGO"""
    #     n_obj = train_obj.shape[-1]
    #
    #     # assume first outcomes of the model are the objectives, the rest constraints
    #     def objective(Z):
    #         return scalarization(Z[..., :n_obj])
    #
    #     constrained_obj = ConstrainedMCObjective(
    #         objective=objective,
    #         constraints=[lambda Z: Z[..., -1]],  # index the constraint
    #     )
    #     return constrained_obj

    def perform_MOBO(self):
        import time
        import warnings

        warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        self.N_TRIALS = 1 if not self.SMOKE_TEST else 2

        self.MC_SAMPLES = 128 if not self.SMOKE_TEST else 16
        problem = self.problem

        verbose = False

        hvs_qehvi_all = []


        # average over multiple trials
        for trial in range(1, self.N_TRIALS + 1):
            torch.manual_seed(trial)

            hvs_qehvi = []

            # call helper functions to generate initial training data and initialize model
            train_x_qehvi, train_obj_qehvi = self.generate_initial_data(n=3 * (self.problem.dim + 1))
            hv = Hypervolume(ref_point=self.ref_point)

            # compute hypervolume
            mll_qehvi, model_qehvi = self.initialize_model(train_x_qehvi, train_obj_qehvi)

            # compute pareto front
            # is_feas = (train_con_qehvi <= 0).all(dim=-1)
            # feas_train_obj = train_obj_qehvi[is_feas]
            # if feas_train_obj.shape[0] > 0:
            pareto_mask = is_non_dominated(train_obj_qehvi)
            pareto_y = train_obj_qehvi[pareto_mask]
            # compute hypervolume
            volume = hv.compute(pareto_y)
            # else:
            #     volume = 0.0

            hvs_qehvi.append(volume)

            # run N_BATCH rounds of BayesOpt after the initial random batch
            for iteration in range(1, self.N_BATCH + 1):

                t0 = time.time()

                # fit the models

                fit_gpytorch_model(mll_qehvi)

                # define the qEI and qNEI acquisition modules using a QMC sampler

                qehvi_sampler = SobolQMCNormalSampler(num_samples=self.MC_SAMPLES)

                # optimize acquisition functions and get new observations

                new_x_qehvi, new_obj_qehvi = self.optimize_qehvi_and_get_observation(
                    model_qehvi, train_obj_qehvi, qehvi_sampler
                )

                # update training points

                train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
                train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])
                # train_con_qehvi = torch.cat([train_con_qehvi, new_con_qehvi])
                self.train_x_qehvi, self.train_obj_qehvi = train_x_qehvi, train_obj_qehvi
                # update progress
                for hvs_list, train_obj, train_x in zip(
                        (hvs_qehvi,),
                        (train_obj_qehvi,),
                        (train_x_qehvi,)
                ):
                    # compute pareto front

                # if feas_train_obj.shape[0] > 0:
                    pareto_mask = is_non_dominated(train_obj)
                    pareto_y = train_obj[pareto_mask]
                    pareto_x = train_x[pareto_mask]

                    # compute feasible hypervolume
                    volume = hv.compute(pareto_y)
                    # else:
                    #     volume = 0.0
                    hvs_list.append(volume)

                # reinitialize the models so they are ready for fitting on next iteration
                # Note: we find improved performance from not warm starting the model hyperparameters
                # using the hyperparameters from the previous iteration
                mll_qehvi, model_qehvi = self.initialize_model(train_x_qehvi, train_obj_qehvi)

                t1 = time.time()

                # if verbose:
                #     print(
                #         f"\nBatch {iteration:>2}: Hypervolume (random, qParEGO, qEHVI) = "
                #         f"({hvs_random[-1]:>4.2f}), "
                #         f"time = {t1-t0:>4.2f}.", end=""
                #     )
                # else:
                #     print(".", end="")
                # the exact output you're looking for:
                sys.stdout.write('\r')
                sys.stdout.write("trial: " + str(trial) + "/" + str(self.N_TRIALS) +
                                 ": [{:{}}] {:.1f}%".format("=" * iteration, self.N_BATCH, (100 / (self.N_BATCH) * iteration)))
                # sys.stdout.write("trial: " + str(trial) + "/" + str(self.N_TRIALS) + ": [%-",(self.N_BATCH),"s] %d%%" % (
                # '=' * iteration, iteration / self.N_BATCH * 100))
                sys.stdout.flush()

            self.pareto_y = pareto_y
            self.pareto_x = pareto_x

            hvs_qehvi_all.append(hvs_qehvi)
        return hvs_qehvi_all