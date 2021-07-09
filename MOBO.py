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

class MOBO:

    def __init__(self, problem, BATCH_SIZE=3, bounds=[None]):

        self.problem = problem

        self.tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
        self.SMOKE_TEST = os.environ.get("SMOKE_TEST")

        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_RESTARTS = 20 if not self.SMOKE_TEST else 2
        self.RAW_SAMPLES = 1024 if not self.SMOKE_TEST else 4
        self.standard_bounds = torch.zeros(2, self.problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1




    def generate_initial_data(self, n):
        # generate training data
        problem = self.problem
        train_x = draw_sobol_samples(bounds=problem.bounds, n=1, q=n, seed=torch.randint(1000000, (1,)).item()).squeeze(
            0)
        train_obj = problem(train_x)
        # negative values imply feasibility in botorch
        train_con = -problem.evaluate_slack(train_x)
        is_feas = (train_con <= 0).all(dim=-1)
        feas_train_obj = train_obj[is_feas]
        self.ref_point = train_obj.max(0).values#problem.ref_point#feas_train_obj.max(0).values#

        return train_x, train_obj, train_con

    def initialize_model(self, train_x, train_obj, train_con):
        # define models for objective and constraint
        train_y = torch.cat([train_obj, train_con], dim=-1)
        model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def optimize_qehvi_and_get_observation(self, model, train_obj, train_con, sampler):
        """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
        # compute feasible observations
        problem = self.problem
        standard_bounds = self.standard_bounds
        is_feas = (train_con <= 0).all(dim=-1)
        # compute points that are better than the known reference point
        better_than_ref = (train_obj > self.ref_point).all(dim=-1)
        # partition non-dominated space into disjoint rectangles
        partitioning = NondominatedPartitioning(
            ref_point=self.ref_point,
            # use observations that are better than the specified reference point and feasible
            Y=train_obj[better_than_ref & is_feas],
        )
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point.tolist(),  # use known reference point
            partitioning=partitioning,
            sampler=sampler,
            # define an objective that specifies which outcomes are the objectives
            objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
            # specify that the constraint is on the last outcome
            constraints=[lambda Z: Z[..., -1]],
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
            sequential=True,
        )
        # observe new values
        new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
        new_obj = problem(new_x)
        # negative values imply feasibility in botorch
        new_con = -problem.evaluate_slack(new_x)
        return new_x, new_obj, new_con

    def get_constrained_mc_objective(self, train_obj, train_con, scalarization):
        """Initialize a ConstrainedMCObjective for qParEGO"""
        n_obj = train_obj.shape[-1]

        # assume first outcomes of the model are the objectives, the rest constraints
        def objective(Z):
            return scalarization(Z[..., :n_obj])

        constrained_obj = ConstrainedMCObjective(
            objective=objective,
            constraints=[lambda Z: Z[..., -1]],  # index the constraint
        )
        return constrained_obj

    def perform_MOBO(self):
        import time
        import warnings

        warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        self.N_TRIALS = 1 if not self.SMOKE_TEST else 2
        self.N_BATCH = 50 if not self.SMOKE_TEST else 5
        self.MC_SAMPLES = 128 if not self.SMOKE_TEST else 16
        problem = self.problem

        verbose = False

        hvs_qehvi_all = []


        # average over multiple trials
        for trial in range(1, self.N_TRIALS + 1):
            torch.manual_seed(trial)

            hvs_qehvi = []

            # call helper functions to generate initial training data and initialize model
            train_x_qehvi, train_obj_qehvi, train_con_qehvi = self.generate_initial_data(n=2 * (self.problem.dim + 1))
            hv = Hypervolume(ref_point=self.ref_point)

            # compute hypervolume
            mll_qehvi, model_qehvi = self.initialize_model(train_x_qehvi, train_obj_qehvi, train_con_qehvi)

            # compute pareto front
            is_feas = (train_con_qehvi <= 0).all(dim=-1)
            feas_train_obj = train_obj_qehvi[is_feas]
            if feas_train_obj.shape[0] > 0:
                pareto_mask = is_non_dominated(feas_train_obj)
                pareto_y = feas_train_obj[pareto_mask]
                # compute hypervolume
                volume = hv.compute(pareto_y)
            else:
                volume = 0.0

            hvs_qehvi.append(volume)

            # run N_BATCH rounds of BayesOpt after the initial random batch
            for iteration in range(1, self.N_BATCH + 1):

                t0 = time.time()

                # fit the models

                fit_gpytorch_model(mll_qehvi)

                # define the qEI and qNEI acquisition modules using a QMC sampler

                qehvi_sampler = SobolQMCNormalSampler(num_samples=self.MC_SAMPLES)

                # optimize acquisition functions and get new observations

                new_x_qehvi, new_obj_qehvi, new_con_qehvi = self.optimize_qehvi_and_get_observation(
                    model_qehvi, train_obj_qehvi, train_con_qehvi, qehvi_sampler
                )

                # update training points

                train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
                train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])
                train_con_qehvi = torch.cat([train_con_qehvi, new_con_qehvi])
                self.train_x_qehvi, self.train_obj_qehvi, self.train_con_qehvi = train_x_qehvi, train_obj_qehvi, train_con_qehvi
                # update progress
                for hvs_list, train_obj, train_con, train_x in zip(
                        (hvs_qehvi,),
                        (train_obj_qehvi,),
                        (train_con_qehvi,),
                        (train_x_qehvi,)
                ):
                    # compute pareto front
                    is_feas = (train_con <= 0).all(dim=-1)
                    feas_train_obj = train_obj[is_feas]
                    feas_train_x   = train_x[is_feas]

                    if feas_train_obj.shape[0] > 0:
                        pareto_mask = is_non_dominated(feas_train_obj)
                        pareto_y = feas_train_obj[pareto_mask]
                        pareto_x = feas_train_x[pareto_mask]

                        # compute feasible hypervolume
                        volume = hv.compute(pareto_y)
                    else:
                        volume = 0.0
                    hvs_list.append(volume)

                # reinitialize the models so they are ready for fitting on next iteration
                # Note: we find improved performance from not warm starting the model hyperparameters
                # using the hyperparameters from the previous iteration
                mll_qehvi, model_qehvi = self.initialize_model(train_x_qehvi, train_obj_qehvi, train_con_qehvi)

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

                sys.stdout.write("trial: " + str(trial) + "/" + str(self.N_TRIALS) + ": [%-20s] %d%%" % (
                '=' * iteration, iteration / self.N_BATCH * 100))
                sys.stdout.flush()
                sleep(0.25)
            self.pareto_y = pareto_y
            self.pareto_x = pareto_x

            hvs_qehvi_all.append(hvs_qehvi)
        return hvs_qehvi_all