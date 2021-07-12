# Multi-objective-Bayesian-Opt

This repository builds on top of BoTorch to provide a fast way to perform Multi-Objective Bayesian optimization.


Example of usage: 


    def problem(x):
        transl = 1/ np.sqrt(2)
        part1 = (x[0] - transl)**2 + (x[1] - transl)**2
        part2 = (x[0] + transl)**2 + (x[1] + transl)**2
        f1 = 1 - np.exp(-part1)
        f2 = 1 - np.exp(-part2)
        return [f1, f2]

    bounds = np.array([[-2.,-2.],[2.,2.]])
    
    from src.MOBO_reg import MOBO

    MOBO_TRY = MOBO(problem, bounds,minimize=True, N_iteration=25)
    hvs_qehvi_all = MOBO_TRY.perform_MOBO()
    

![alt text](https://github.com/panos108/Multi-objective-Bayesian-Opt/vlmp2.png?raw=true)



Requirements:

Python >= 3.6

PyTorch >= 1.6

gpytorch >= 1.5

Botorch 

scipy

