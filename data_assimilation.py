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
import torch 

import os
import sys
import time

from L96_emulator.util import dtype, dtype_np, device
from L96_emulator.util import sortL96fromChannels, sortL96intoChannels
from L96_emulator.util import predictor_corrector, rk4_default, get_data

from L96_emulator.run import setup, sel_dataset_class

from L96_emulator.eval import load_model_from_exp_conf, get_rollout_fun, optim_initial_state, named_network

import L96sim
from L96sim.L96_base import f1, f2, pf2

res_dir = '/gpfs/work/nonnenma/results/emulators/L96/'
data_dir = '/gpfs/work/nonnenma/data/emulators/L96/'

K, J, T, dt, N_trials = 36, 10, 605, 0.01, 1
spin_up_time, train_frac = 5., 0.8
normalize_data = False
F, h, b, c = 10., 1., 10., 10.

lead_time = 1
prediction_task = 'state'

exp_id = 20
model_forwarder = 'rk4_default'
dt_net = dt

if exp_id is None: 
    # loading 'perfect' (up to machine-precision-level quirks) L96 model in pytorch
    
    conf_exp = '00_analyticalMinimalConvNet'
    args = {'filters': [0],
           'kernel_sizes': [4],
           'init_net': 'analytical',
           'K_net': K,
           'J_net': J,
           'dt_net': dt_net,
           'model_forwarder': model_forwarder}
    model, model_forwarder = named_network(
        model_name='MinimalConvNetL96',
        n_input_channels=J+1,
        n_output_channels=J+1,
        seq_length=1,
        **args
    )
else:

    exp_names = os.listdir('experiments/')   
    conf_exp = exp_names[np.where(np.array([name[:2] for name in exp_names])==str(exp_id))[0][0]][:-4]
    print('conf_exp', conf_exp)

    # ### pick a (trained) emulator

    args = setup(conf_exp=f'experiments/{conf_exp}.yml')
    args.pop('conf_exp')

    # ### choose numerical solver scheme

    args['model_forwarder'] = model_forwarder
    args['dt_net'] = dt_net

    # ### load & instantiate the emulator

    model, model_forwarder, _ = load_model_from_exp_conf(res_dir, args)

# switch off parameter gradients for model:
for x in model.parameters():
    x.requires_grad = False

n_starts = np.arange(int(spin_up_time/dt), int(train_frac*T/dt), 2* int(spin_up_time/dt))
T_rollout, N = 40, len(n_starts)
n_chunks = 40
n_steps = 1000 # total number of gradient steps (across all chunks !)

back_solve_dt_fac = 100

lbfgs_pars = {'n_steps' : n_steps//n_chunks,
              'lr' : 1.0,
              'max_iter' : 10000,
              'max_eval' : None,
              'tolerance_grad' : 1e-07, 
              'tolerance_change' : 1e-09, 
              'history_size': 100}


# # Solving a fully-observed inverse problem

# ### load / simulate data

out, datagen_dict = get_data(K=K, J=J, T=T, dt=dt, N_trials=N_trials, F=F, h=h, b=b, c=c, 
                             resimulate=True, solver=rk4_default,
                             save_sim=False, data_dir=data_dir)

DatasetClass = sel_dataset_class(prediction_task=prediction_task,N_trials=1)
dg_train = DatasetClass(data=out, J=J, offset=lead_time, normalize=normalize_data, 
                   start=int(spin_up_time/dt), 
                   end=int(np.floor(out.shape[0]*train_frac)))


# ## L-BFGS, split rollout time into chunks, solve sequentially from end to beginning

print('\n')
print('L-BFGS, split rollout time into chunks, solve sequentially from end to beginning')
print('\n')

T_rollouts = np.ones(n_chunks, dtype=np.int) * (T_rollout//n_chunks)
x_inits, targets = [None for i in range(n_chunks)], [None for i in range(n_chunks)]
x_inits[0] = sortL96intoChannels(np.atleast_2d(out[n_starts+T_rollout].copy()),J=J)
targets[0] = out[n_starts+T_rollout].copy()
grndtrths = [out[n_starts+T_rollout-(j+1)*(T_rollout//n_chunks)] for j in range(n_chunks)]

res = optim_initial_state(
      model_forwarder, K, J, N,
      n_steps, lbfgs_pars,
      x_inits, targets, grndtrths,
      out, n_starts, T_rollouts, n_chunks)

x_sols_LBFGS_chunks, loss_vals_LBFGS_chunks, time_vals_LBFGS_chunks, state_mses_LBFGS_chunks = res

# ## L-BFGS, solve across full rollout time in one go, initialize from chunked approach

print('\n')
print('L-BFGS, solve across full rollout time in one go, initialize from chunked approach')
print('\n')

loss_vals_LBFGS_full_chunks = np.zeros(n_steps)


T_rollouts = np.arange(1, n_chunks+1) * (T_rollout//n_chunks)
x_inits = [sortL96intoChannels(z,J=J).copy() for z in x_sols_LBFGS_chunks] 
targets = [out[n_starts+T_rollout].copy() for i in range(n_chunks)]

res = optim_initial_state(
      model_forwarder, K, J, N,
      n_steps, lbfgs_pars,
      x_inits, targets, grndtrths,
      out, n_starts, T_rollouts, n_chunks)

x_sols_LBFGS_full_chunks, loss_vals_LBFGS_full_chunks, time_vals_LBFGS_full_chunks, state_mses_LBFGS_full_chunks = res


# ## L-BFGS, solve across full rollout time in one go, initiate from backward solution

print('\n')
print('L-BFGS, solve across full rollout time in one go, initiate from backward solution')
print('\n')

dX_dt = np.empty(K*(J+1), dtype=dtype_np)
if J > 0:
    def fun(t, x):
        return - f2(x, F, h, b, c, dX_dt, K, J)
else:
    def fun(t, x):
        return - f1(x, F, dX_dt, K)
state_mses_backsolve = np.zeros(n_chunks)
time_vals_backsolve = np.zeros(n_chunks)
i_ = 0
for j in range(n_chunks):
    
    T_i = T_rollouts[j]
    times = dt * np.linspace(0, T_i, back_solve_dt_fac * T_i+1) # note the increase in temporal resolution!
    print('backward solving')
    time_vals_backsolve[j] = time.time()
    for i__ in range(len(n_starts)):
        out2 = rk4_default(fun=fun, y0=out[n_starts[i__]+T_rollout].copy(), times=times)
        x_init[i__] = out2[-1].copy()
    x_sols_backsolve[j] = x_init.copy()
    time_vals_backsolve[j] = time.time() - time_vals_backsolve[j]
    state_mses_backsolve[j] = ((x_init - out[n_starts])**2).mean()

x_inits = [sortL96intoChannels(z,J=J).copy() for z in x_sols_backsolve]
res = optim_initial_state(
      model_forwarder, K, J, N,
      n_steps, lbfgs_pars,
      x_inits, targets, grndtrths,
      out, n_starts, T_rollouts, n_chunks)

x_sols_LBFGS_full_backsolve, loss_vals_LBFGS_full_backsolve, time_vals_LBFGS_full_backsolve, state_mses_full_backsolve = res


# ## L-BFGS, solve across full rollout time in one go
# - warning, this can be excruciatingly slow and hard to converge !

print('\n')
print('L-BFGS, solve across full rollout time in one go')
print('\n')

x_init = sortL96intoChannels(out[n_starts+T_rollout], J=J)
x_inits = [x_init.copy() for j in range(n_chunks)]
res = optim_initial_state(
      model_forwarder, K, J, N,
      n_steps, lbfgs_pars,
      x_inits, targets, grndtrths,
      out, n_starts, T_rollouts, n_chunks)

x_sols_LBFGS_full_persistence, loss_vals_LBFGS_full_persistence, time_vals_full_persistence, state_mses_full_persistence = res


# ## plot and compare results

print('\n')
print('done, storing results')
print('\n')

initial_states = [out[n_starts+j*T_rollout//n_chunks] for j in range(n_chunks)]
initial_states = np.stack([sortL96intoChannels(z,J=J) for z in initial_states])

model_forwarder_str = args['model_forwarder']
np.save(res_dir + f'results/data_assimilation/fullyobs_initstate_tests_exp{exp_id}_{model_forwarder_str}',
        arr={
            'K' : K,
            'J' : J,
            'T' : T, 
            'dt' : dt,
            
            'spin_up_time' : spin_up_time,
            'train_frac' : train_frac,
            'normalize_data' : normalize_data,
            'F' : F, 
            'h' : h, 
            'b' : b, 
            'c' : c,
            'lead_time' : lead_time,

            'conf_exp' : conf_exp,
            'model_forwarder' : args['model_forwarder'], # should still be string
            'dt_net' : args['dt_net'],
            'back_solve_dt_fac' : back_solve_dt_fac,
            
            'n_starts' : n_starts,
            'T_rollout' : T_rollout,
            'n_chunks' : n_chunks,
            'n_steps' : n_steps,
                        
            'lbfgs_pars' : lbfgs_pars, 

            'targets' : sortL96intoChannels(out[n_starts+T_rollout], J=J),
            'initial_states' : initial_states,
            
            'loss_vals_LBFGS_full_backsolve' : loss_vals_LBFGS_full_backsolve, 
            'loss_vals_LBFGS_full_persistence' : loss_vals_LBFGS_full_persistence,
            'loss_vals_LBFGS_full_chunks' : loss_vals_LBFGS_full_chunks,
            'loss_vals_LBFGS_chunks' : loss_vals_LBFGS_chunks,
            'loss_vals_LBFGS_chunks_rollout' : loss_vals_LBFGS_chunks_rollout,

            'time_vals_LBFGS_full_backsolve' :   time_vals_LBFGS_full_backsolve,
            'time_vals_LBFGS_full_persistence' : time_vals_LBFGS_full_persistence,
            'time_vals_LBFGS_full_chunks' :      time_vals_LBFGS_full_chunks,
            'time_vals_LBFGS_chunks' :           time_vals_LBFGS_chunks,
            'time_vals_backsolve' :              time_vals_backsolve,

            'state_mses_LBFGS_full_backsolve' :   state_mses_LBFGS_full_backsolve,
            'state_mses_LBFGS_full_persistence' : state_mses_LBFGS_full_persistence,
            'state_mses_LBFGS_full_chunks' :      state_mses_LBFGS_full_chunks,
            'state_mses_LBFGS_chunks' :           state_mses_LBFGS_chunks,
            'state_mses_backsolve' :              state_mses_backsolve,
            
            'x_sols_LBFGS_full_backsolve' : x_sols_LBFGS_full_backsolve, 
            'x_sols_LBFGS_full_persistence' : x_sols_LBFGS_full_persistence,
            'x_sols_LBFGS_full_chunks' : x_sols_LBFGS_full_chunks,
            'x_sols_LBFGS_chunks' : x_sols_LBFGS_chunks,
            'x_sols_backsolve' : x_sols_backsolve
            })
