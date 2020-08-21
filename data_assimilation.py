#!/usr/bin/env python
# coding: utf-8

# # 1.3 Solve a fully-observed inverse problem
# 
# Given $x_T$, estimate $x_0$ by matching $G^{(T)}(x_0)$ to $x_T$. Use autodiff on $G$ to calculate gradients of an error metric w.r.t. $x_0$. Compare the resulting rollout to the original `true' simulation.
# 
# Compare three approaches:
# - $argmin_{x_0} || x_T - G^{(T)}(x_i) ||$, i.e. $T$ steps in one go
# - $argmin_{x_i} || x_{i+T_i} - G^{(T_i)}(x_i) ||$, i.e. $T_i$ steps at a time, with $\sum_i T_i = T$. In the extreme case of $T_i=1$, this becomes very similar to implicit numerical methods. Can invertible neural networks help beyond providing better initializations for $x_i$ ? 
# - solving backwards: more of the extreme case of $\forall i: T_i=1$, however: Only for some forward numerical solvers can we just reverse time [1] and expect to return to initial conditions. Leap-frog works, but e.g. forward-Euler time-reversed is backward-Euler. 
# 
# Generally, how do these approaches differ around \& beyond the horizon of predictability? Which solutions do they pick, and how easy is it to get uncertainties from them?
# 
# [1] https://scicomp.stackexchange.com/questions/32736/forward-and-backward-integration-cause-of-errors?noredirect=1&lq=1


import numpy as np

from L96_emulator.util import dtype, dtype_np, device
from L96_emulator.data_assimilation import solve_initstate, ObsOp_subsampleGaussian, ObsOp_identity

res_dir = '/gpfs/work/nonnenma/results/emulators/L96/'
data_dir = '/gpfs/work/nonnenma/data/emulators/L96/'

system_pars = {
    'K' : 36,
    'J' : 0,
    'T' : 605,
    'dt' : 0.05,
    'back_solve_dt_fac' : 1000,
    'N_trials' : 1,
    'spin_up_time' : 5.,
    'train_frac' : 0.8,
    'normalize_data' : False,
    'F' : 10.,
    'h' : 1.,
    'b' : 10.,
    'c' : 10.,
    'obs_operator' : ObsOp_identity, #ObsOp_identity, #ObsOp_subsampleGaussian,
    'obs_operator_args' : {} #{'r' : 0.0, 'sigma2' : 1.0} #{} #{'r' : 0.5, 'sigma2' : 1.0}
}

setup_pars = {
    'n_starts' : np.arange(int(system_pars['spin_up_time']/system_pars['dt']),
                           int(system_pars['train_frac']*system_pars['T']/system_pars['dt']),
                           8*3* int(system_pars['spin_up_time']/system_pars['dt'])),
    'T_rollout' : 40,          # number of rollout steps (by model_forwarder, e.g. RK4 step)
    'n_chunks' : 8,            # number of intermediate chunks to try solving for initial state
    'n_chunks_recursive' : 40, # for recursive methods (such as solving forward in reverse), can give more chunks
    'prediction_task' : 'state',
    'lead_time' : 1
}

model_pars = {
    'exp_id' : 24,
    'model_forwarder' : 'rk4_default',
    'K_net' : system_pars['K'],
    'J_net' : system_pars['J'],
    'dt_net' : system_pars['dt']
}

optimizer_pars = {
              'optimizer' : 'LBFGS',
              'n_steps' : 50,
              'lr' : 1e0,
              'max_iter' : 1000,
              'max_eval' : None,
              'tolerance_grad' : 1e-07,
              'tolerance_change' : 1e-09,
              'history_size': 10
}

optimiziation_schemes = {
    'LBFGS_chunks' : True,
    'LBFGS_full_chunks' : True,
    'backsolve' : True, 
    'LBFGS_full_backsolve' : True,
    'LBFGS_full_persistence' : True, 
    'LBFGS_recurse_chunks' : True
}

solve_initstate(system_pars=system_pars,
                model_pars=model_pars,
                optimizer_pars=optimizer_pars,
                setup_pars=setup_pars,
                optimiziation_schemes=optimiziation_schemes,
                res_dir=res_dir,
                data_dir=data_dir)
