import os
import torch
import numpy as np
np.random.seed(0)
import random
random.seed(0)
torch.manual_seed(0)


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

from botorch.test_functions.multi_objective import C2DTLZ2


d = 12
M = 2
problem = C2DTLZ2(dim=d, num_objectives=M, negate=True).to(**tkwargs)

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples


def generate_initial_data(n):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds,n=1, q=n, seed=torch.randint(1000000, (1,)).item()).squeeze(0)
    train_obj = problem(train_x)
    # negative values imply feasibility in botorch
    train_con = -problem.evaluate_slack(train_x)
    ref_point = train_obj.max(0).values

    return train_x, train_obj, train_con, ref_point


def initialize_model(train_x, train_obj, train_con):
    # define models for objective and constraint
    train_y = torch.cat([train_obj, train_con], dim=-1)
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model



from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex


BATCH_SIZE = 2
NUM_RESTARTS = 20 if not SMOKE_TEST else 2
RAW_SAMPLES = 1024 if not SMOKE_TEST else 4

standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
standard_bounds[1] = 1


def optimize_qehvi_and_get_observation(model, train_obj, train_con, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # compute feasible observations
    is_feas = (train_con <= 0).all(dim=-1)
    # compute points that are better than the known reference point
    better_than_ref = (train_obj > ref_point).all(dim=-1)
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(
        ref_point=ref_point,
        # use observations that are better than the specified reference point and feasible
        Y=train_obj[better_than_ref & is_feas],
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),  # use known reference point
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
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    # observe new values
    new_x =  unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj = problem(new_x)
    # negative values imply feasibility in botorch
    new_con = -problem.evaluate_slack(new_x)
    return new_x, new_obj, new_con



from botorch.acquisition.objective import ConstrainedMCObjective


def get_constrained_mc_objective(train_obj,train_con,scalarization):
    """Initialize a ConstrainedMCObjective for qParEGO"""
    n_obj = train_obj.shape[-1]
    # assume first outcomes of the model are the objectives, the rest constraints
    def objective(Z):
        return scalarization(Z[..., :n_obj])

    constrained_obj = ConstrainedMCObjective(
        objective=objective,
        constraints=[lambda Z: Z[..., -1]], # index the constraint
    )
    return constrained_obj




def optimize_qparego_and_get_observation(model, train_obj, train_con, sampler):
    """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization
    of the qParEGO acquisition function, and returns a new candidate and observation."""
    acq_func_list = []
    for _ in range(BATCH_SIZE):
        # sample random weights
        weights = sample_simplex(problem.num_objectives, **tkwargs).squeeze()
        # construct augmented Chebyshev scalarization
        scalarization = get_chebyshev_scalarization(weights=weights, Y=train_obj)
        # initialize ConstrainedMCObjective
        constrained_objective = get_constrained_mc_objective(
            train_obj=train_obj, train_con=train_con, scalarization=scalarization,
        )
        train_y = torch.cat([train_obj, train_con], dim=-1)
        acq_func = qExpectedImprovement(  # pyre-ignore: [28]
            model=model,
            objective=constrained_objective,
            best_f=constrained_objective(train_y).max(),
            sampler=sampler,
        )
        acq_func_list.append(acq_func)
    # optimize
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds=standard_bounds,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj = problem(new_x)
    # negative values imply feasibility in botorch
    new_con = -problem.evaluate_slack(new_x)
    return new_x, new_obj, new_con


from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

import time
import warnings

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

N_TRIALS = 1 if not SMOKE_TEST else 2
N_BATCH = 50 if not SMOKE_TEST else 5
MC_SAMPLES = 128 if not SMOKE_TEST else 16

verbose = False

hvs_qparego_all, hvs_qehvi_all, hvs_random_all = [], [], []


# average over multiple trials
for trial in range(1, N_TRIALS + 1):
    torch.manual_seed(trial)

    print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
    hvs_qparego, hvs_qehvi, hvs_random = [], [], []

    # call helper functions to generate initial training data and initialize model
    train_x_qparego, train_obj_qparego, train_con_qparego, ref_point = generate_initial_data(n=2 * (d + 1))
    mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego, train_con_qparego)

    train_x_qehvi, train_obj_qehvi, train_con_qehvi = train_x_qparego, train_obj_qparego, train_con_qparego
    train_x_random, train_obj_random, train_con_random = train_x_qparego, train_obj_qparego, train_con_qparego
    # compute hypervolume
    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, train_con_qehvi)
    hv = Hypervolume(ref_point=ref_point)

    # compute pareto front
    is_feas = (train_con_qparego <= 0).all(dim=-1)
    feas_train_obj = train_obj_qparego[is_feas]
    if feas_train_obj.shape[0] > 0:
        pareto_mask = is_non_dominated(feas_train_obj)
        pareto_y = feas_train_obj[pareto_mask]
        # compute hypervolume
        volume = hv.compute(pareto_y)
    else:
        volume = 0.0

    hvs_qparego.append(volume)
    hvs_qehvi.append(volume)
    hvs_random.append(volume)

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):

        t0 = time.time()

        # fit the models
        fit_gpytorch_model(mll_qparego)
        fit_gpytorch_model(mll_qehvi)

        # define the qEI and qNEI acquisition modules using a QMC sampler
        qparego_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
        qehvi_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

        # optimize acquisition functions and get new observations
        new_x_qparego, new_obj_qparego, new_con_qparego = optimize_qparego_and_get_observation(
            model_qparego, train_obj_qparego, train_con_qparego, qparego_sampler
        )
        new_x_qehvi, new_obj_qehvi, new_con_qehvi = optimize_qehvi_and_get_observation(
            model_qehvi, train_obj_qehvi, train_con_qehvi, qehvi_sampler
        )
        new_x_random, new_obj_random, new_con_random, _ = generate_initial_data(n=BATCH_SIZE)

        # update training points
        train_x_qparego = torch.cat([train_x_qparego, new_x_qparego])
        train_obj_qparego = torch.cat([train_obj_qparego, new_obj_qparego])
        train_con_qparego = torch.cat([train_con_qparego, new_con_qparego])

        train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
        train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])
        train_con_qehvi = torch.cat([train_con_qehvi, new_con_qehvi])

        train_x_random = torch.cat([train_x_random, new_x_random])
        train_obj_random = torch.cat([train_obj_random, new_obj_random])
        train_con_random = torch.cat([train_con_random, new_con_random])

        # update progress
        for hvs_list, train_obj, train_con in zip(
                (hvs_random, hvs_qparego, hvs_qehvi),
                (train_obj_random, train_obj_qparego, train_obj_qehvi),
                (train_con_random, train_con_qparego, train_con_qehvi),
        ):
            # compute pareto front
            is_feas = (train_con <= 0).all(dim=-1)
            feas_train_obj = train_obj[is_feas]
            if feas_train_obj.shape[0] > 0:
                pareto_mask = is_non_dominated(feas_train_obj)
                pareto_y = feas_train_obj[pareto_mask]
                # compute feasible hypervolume
                volume = hv.compute(pareto_y)
            else:
                volume = 0.0
            hvs_list.append(volume)

        # reinitialize the models so they are ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration
        mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego, train_con_qparego)
        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, train_con_qehvi)

        t1 = time.time()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: Hypervolume (random, qParEGO, qEHVI) = "
                f"({hvs_random[-1]:>4.2f}, {hvs_qparego[-1]:>4.2f}, {hvs_qehvi[-1]:>4.2f}), "
                f"time = {t1 - t0:>4.2f}.", end=""
            )
        else:
            print(".", end="")

    hvs_qparego_all.append(hvs_qparego)
    hvs_qehvi_all.append(hvs_qehvi)
    hvs_random_all.append(hvs_random)

import numpy as np
from matplotlib import pyplot as plt




def ci(y):
    return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)


iters = np.arange(N_BATCH + 1) * BATCH_SIZE
log_hv_difference_qparego = np.log10(problem.max_hv - np.asarray(hvs_qparego_all))
log_hv_difference_qehvi = np.log10(problem.max_hv - np.asarray(hvs_qehvi_all))
log_hv_difference_rnd = np.log10(problem.max_hv - np.asarray(hvs_random_all))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.errorbar(
    iters, log_hv_difference_rnd.mean(axis=0), yerr=ci(log_hv_difference_rnd),
    label="Sobol", linewidth=1.5,
)
ax.errorbar(
    iters, log_hv_difference_qparego.mean(axis=0), yerr=ci(log_hv_difference_qparego),
    label="qParEGO", linewidth=1.5,
)
ax.errorbar(
    iters, log_hv_difference_qehvi.mean(axis=0), yerr=ci(log_hv_difference_qehvi),
    label="qEHVI", linewidth=1.5,
)
ax.set(xlabel='number of observations (beyond initial points)', ylabel='Log Hypervolume Difference')
ax.legend(loc="lower right")

from matplotlib.cm import ScalarMappable

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
algos = ["Sobol", "qParEGO", "qEHVI"]
cm = plt.cm.get_cmap('viridis')

batch_number = torch.cat(
    [torch.zeros(2 * (d + 1)), torch.arange(1, N_BATCH + 1).repeat(BATCH_SIZE, 1).t().reshape(-1)]
).numpy()

for i, train_obj in enumerate((train_obj_random, train_obj_qparego, train_obj_qehvi)):
    sc = axes[i].scatter(train_obj[:, 0].cpu().numpy(), train_obj[:, 1].cpu().numpy(), c=batch_number, alpha=0.8)
    axes[i].set_title(algos[i])
    axes[i].set_xlabel("Objective 1")
    axes[i].set_xlim(-2.5, 0)
    axes[i].set_ylim(-2.5, 0)
axes[0].set_ylabel("Objective 2")
norm = plt.Normalize(batch_number.min(), batch_number.max())
sm = ScalarMappable(norm=norm, cmap=cm)
sm.set_array([])
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title("Iteration")