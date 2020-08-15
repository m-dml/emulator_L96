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

from L96_emulator.eval import load_model_from_exp_conf, get_rollout_fun, Rollout

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

exp_names = os.listdir('experiments/')   
conf_exp = exp_names[np.where(np.array([name[:2] for name in exp_names])==str(exp_id))[0][0]][:-4]
print('conf_exp', conf_exp)

model_forwarder = 'rk4_default'
dt_net = dt

n_starts = np.arange(int(spin_up_time/dt), int(train_frac*T/dt), 2* int(spin_up_time/dt))
T_rollout, N = 40, len(n_starts)
n_chunks = 10
n_steps = 1000 # total number of gradient steps (across all chunks !)

back_solve_dt_fac = 100

lbfgs_pars = {'n_steps' : n_steps//n_chunks,
              'lr' : 1.0,
              'max_iter' : 100,
              'max_eval' : None,
              'tolerance_grad' : 1e-07, 
              'tolerance_change' : 1e-09, 
              'history_size': 50,
              'line_search_fn': 'strong_wolfe'} 


# # Solving a fully-observed inverse problem

# ### load / simulate data

out, datagen_dict = get_data(K=K, J=J, T=T, dt=dt, N_trials=N_trials, F=F, h=h, b=b, c=c, 
                             resimulate=True, solver=rk4_default,
                             save_sim=False, data_dir=data_dir)

DatasetClass = sel_dataset_class(prediction_task=prediction_task,N_trials=1)
dg_train = DatasetClass(data=out, J=J, offset=lead_time, normalize=normalize_data, 
                   start=int(spin_up_time/dt), 
                   end=int(np.floor(out.shape[0]*train_frac)))


# ### pick a (trained) emulator

args = setup(conf_exp=f'experiments/{conf_exp}.yml')
args.pop('conf_exp')

# ### choose numerical solver scheme

args['model_forwarder'] = model_forwarder
args['dt_net'] = dt_net

# ### load & instantiate the emulator

model, model_forwarder, training_outputs = load_model_from_exp_conf(res_dir, args)

# ## L-BFGS, split rollout time into chunks, solve sequentially from end to beginning

loss_vals_LBFGS_chunks = np.zeros(n_steps)
time_vals_LBFGS_chunks = time.time() * np.ones(n_steps)
loss_vals_LBFGS_chunks_rollout = np.zeros_like(loss_vals_LBFGS_chunks)

x_sols_LBFGS_chunks = np.zeros((n_chunks, N, K*(J+1)))
state_mses_LBFGS_chunks = np.zeros(n_chunks)

grndtrth_rollout = sortL96intoChannels(torch.as_tensor(out[n_starts], dtype=dtype, device=device), J=J)
target_rollout = sortL96intoChannels(torch.as_tensor(out[n_starts+T_rollout], dtype=dtype, device=device),J=J)

x_inits = np.zeros((n_chunks, N, K*(J+1)))
x_init = sortL96intoChannels(np.atleast_2d(out[n_starts+T_rollout].copy()),J=J)
targets = np.zeros((n_chunks, N, K*(J+1)))
targets[0] = out[n_starts+T_rollout]

i_ = 0
for j in range(n_chunks):
    roller_outer_LBFGS_chunks = Rollout(model_forwarder, prediction_task='state', K=K, J=J, 
                                        N=N, T=(T_rollout//n_chunks), 
                                        x_init=x_init)
    x_inits[j] = sortL96fromChannels(roller_outer_LBFGS_chunks.X.detach().cpu().numpy().copy())
    optimizer = torch.optim.LBFGS(params=roller_outer_LBFGS_chunks.parameters(), 
                                  lr=lbfgs_pars['lr'], 
                                  max_iter=lbfgs_pars['max_iter'], 
                                  max_eval=lbfgs_pars['max_eval'], 
                                  tolerance_grad=lbfgs_pars['tolerance_grad'], 
                                  tolerance_change=lbfgs_pars['tolerance_change'], 
                                  history_size=lbfgs_pars['history_size'], 
                                  line_search_fn=lbfgs_pars['line_search_fn'])

    target = sortL96intoChannels(torch.as_tensor(targets[j], dtype=dtype, device=device),J=J)
    roller_outer_LBFGS_chunks.train()
    for i in range(n_steps//n_chunks):

        loss = ((roller_outer_LBFGS_chunks.forward() - target)**2).mean()
        
        if torch.isnan(loss):
            i_ += 1
            continue

        def closure():
            loss = ((roller_outer_LBFGS_chunks.forward() - target)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            return loss            
        optimizer.step(closure)        
        loss_vals_LBFGS_chunks[i_] = loss.detach().cpu().numpy()
        time_vals_LBFGS_chunks[i_] = time.time() - time_vals_LBFGS_chunks[i_]
        print((time_vals_LBFGS_chunks[i_], loss_vals_LBFGS_chunks[i_]))

        roller_outer_LBFGS_chunks.T = (j+1)*T_rollout//n_chunks
        loss = ((roller_outer_LBFGS_chunks.forward() - target_rollout)**2).mean()
        loss_vals_LBFGS_chunks_rollout[i_] = loss.detach().cpu().numpy().copy()

        i_ += 1

    x_init = roller_outer_LBFGS_chunks.X.detach().cpu().numpy().copy()
    if j < n_chunks - 1:
        targets[j+1] = sortL96fromChannels(roller_outer_LBFGS_chunks.X.detach().cpu().numpy().copy())

    x_sols_LBFGS_chunks[j] = sortL96fromChannels(roller_outer_LBFGS_chunks.X.detach().cpu().numpy().copy())
    grndtrth = out[n_starts+T_rollout-(j+1)*(T_rollout//n_chunks)]
    state_mses_LBFGS_chunks[j] = ((x_sols_LBFGS_chunks[j] - grndtrth)**2).mean()


# ## L-BFGS, solve across full rollout time in one go
# - warning, this can be excruciatingly slow and hard to converge !

loss_vals_LBFGS_full_persistence = np.zeros(n_steps)
time_vals_LBFGS_full_persistence = time.time() * np.ones(n_steps)

x_sols_LBFGS_full_persistence = np.zeros((n_chunks, N, K*(J+1)))
state_mses_LBFGS_full_persistence = np.zeros(n_chunks)

x_init = sortL96intoChannels(out[n_starts+T_rollout].copy(), J=J)
target = sortL96intoChannels(torch.as_tensor(out[n_starts+T_rollout], dtype=dtype, device=device), J=J)

i_ = 0
for j in range(n_chunks):

    roller_outer_LBFGS_full_persistence = Rollout(model_forwarder, prediction_task='state', K=K, J=J, 
                                        N=N, T=(j+1)*T_rollout//n_chunks, x_init=x_init)
    optimizer = torch.optim.LBFGS(params=roller_outer_LBFGS_full_persistence.parameters(),
                                  lr=lbfgs_pars['lr'], 
                                  max_iter=lbfgs_pars['max_iter'], 
                                  max_eval=lbfgs_pars['max_eval'], 
                                  tolerance_grad=lbfgs_pars['tolerance_grad'], 
                                  tolerance_change=lbfgs_pars['tolerance_change'], 
                                  history_size=lbfgs_pars['history_size'], 
                                  line_search_fn=lbfgs_pars['line_search_fn'])

    roller_outer_LBFGS_full_persistence.train()
    for i in range(n_steps//n_chunks):

        loss = ((roller_outer_LBFGS_full_persistence.forward() - target)**2).mean()
        
        if torch.isnan(loss):
            i_ += 1
            continue

        def closure():
            loss = ((roller_outer_LBFGS_full_persistence.forward() - target)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            return loss            
        optimizer.step(closure)        
        loss_vals_LBFGS_full_persistence[i_] = loss.detach().cpu().numpy()
        time_vals_LBFGS_full_persistence[i_] = time.time() - time_vals_LBFGS_full_persistence[i_]
        print((time_vals_LBFGS_full_persistence[i_], loss_vals_LBFGS_full_persistence[i_]))
        i_ += 1

    x_sols_LBFGS_full_persistence[j] = sortL96fromChannels(roller_outer_LBFGS_full_persistence.X.detach().cpu().numpy().copy())
    grndtrth = out[n_starts+T_rollout-(j+1)*(T_rollout//n_chunks)]
    state_mses_LBFGS_full_persistence[j] = ((x_sols_LBFGS_full_persistence[j] - grndtrth)**2).mean()


# ## L-BFGS, solve across full rollout time in one go, initialize from chunked approach

loss_vals_LBFGS_full_chunks = np.zeros(n_steps)
time_vals_LBFGS_full_chunks = time.time() * np.ones(n_steps)

x_sols_LBFGS_full_chunks = np.zeros((n_chunks, N, K*(J+1)))
state_mses_LBFGS_full_chunks = np.zeros(n_chunks)

target = sortL96intoChannels(torch.as_tensor(out[n_starts+T_rollout], dtype=dtype, device=device), J=J)

i_ = 0
for j in range(n_chunks):

    x_init = sortL96intoChannels(x_sols_LBFGS_chunks[j], J=J)
    roller_outer_LBFGS_full_chunks = Rollout(model_forwarder, prediction_task='state', K=K, J=J, 
                                        N=N, T=(j+1)*T_rollout//n_chunks, x_init=x_init)
    optimizer = torch.optim.LBFGS(params=roller_outer_LBFGS_full_chunks.parameters(), 
                                  lr=lbfgs_pars['lr'], 
                                  max_iter=lbfgs_pars['max_iter'], 
                                  max_eval=lbfgs_pars['max_eval'], 
                                  tolerance_grad=lbfgs_pars['tolerance_grad'], 
                                  tolerance_change=lbfgs_pars['tolerance_change'], 
                                  history_size=lbfgs_pars['history_size'], 
                                  line_search_fn=lbfgs_pars['line_search_fn'])

    roller_outer_LBFGS_full_chunks.train()
    for i in range(n_steps//n_chunks):

        loss = ((roller_outer_LBFGS_full_chunks.forward() - target)**2).mean()
        
        if torch.isnan(loss):
            i_ += 1
            continue

        def closure():
            loss = ((roller_outer_LBFGS_full_chunks.forward() - target)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            return loss            
        optimizer.step(closure)        
        loss_vals_LBFGS_full_chunks[i_] = loss.detach().cpu().numpy()
        time_vals_LBFGS_full_chunks[i_] = time.time() - time_vals_LBFGS_full_chunks[i_]
        print((time_vals_LBFGS_full_chunks[i_], loss_vals_LBFGS_full_chunks[i_]))
        i_ += 1

    x_sols_LBFGS_full_chunks[j] = sortL96fromChannels(roller_outer_LBFGS_full_chunks.X.detach().cpu().numpy().copy())
    grndtrth = out[n_starts+T_rollout-(j+1)*(T_rollout//n_chunks)]
    state_mses_LBFGS_full_chunks[j] = ((x_sols_LBFGS_full_chunks[j] - grndtrth)**2).mean()


# ## L-BFGS, solve across full rollout time in one go, initiate from backward solution

dX_dt = np.empty(K*(J+1), dtype=dtype_np)
if J > 0:
    def fun(t, x):
        return - f2(x, F, h, b, c, dX_dt, K, J)
else:
    def fun(t, x):
        return - f1(x, F, dX_dt, K)

loss_vals_LBFGS_full_backsolve = np.zeros(n_steps)
time_vals_LBFGS_full_backsolve = time.time() * np.ones(n_steps)

x_sols_backsolve = np.zeros((n_chunks, N, K*(J+1)))
x_sols_LBFGS_full_backsolve = np.zeros((n_chunks, N, K*(J+1)))
state_mses_LBFGS_full_backsolve = np.zeros(n_chunks)

target = sortL96intoChannels(torch.as_tensor(out[n_starts+T_rollout], dtype=dtype, device=device), J=J)
x_init = np.zeros((len(n_starts), K*(J+1)))

state_mses_backsolve = np.zeros(n_chunks)
time_vals_backsolve = np.zeros(n_chunks)

i_ = 0
for j in range(n_chunks):
    
    T_i = (j+1)*T_rollout//n_chunks
    times = dt * np.linspace(0, T_i, back_solve_dt_fac * T_i+1) # note the increase in temporal resolution!
    print('backward solving')
    time_vals_backsolve[j] = time.time()
    for i__ in range(len(n_starts)):
        out2 = rk4_default(fun=fun, y0=out[n_starts[i__]].copy(), times=times)
        x_init[i__] = out2[-1].copy()
    x_sols_backsolve[j] = x_init.copy()
    time_vals_backsolve[j] = time.time() - time_vals_backsolve[j]
    state_mses_backsolve[j] = ((x_init - out[n_starts])**2).mean()

    roller_outer_LBFGS_full_backsolve = Rollout(model_forwarder, prediction_task='state', K=K, J=J, 
                                        N=N, T=(j+1)*T_rollout//n_chunks, 
                                        x_init=sortL96intoChannels(x_init,J=J))
    optimizer = torch.optim.LBFGS(params=roller_outer_LBFGS_full_backsolve.parameters(), 
                                  lr=lbfgs_pars['lr'], 
                                  max_iter=lbfgs_pars['max_iter'], 
                                  max_eval=lbfgs_pars['max_eval'], 
                                  tolerance_grad=lbfgs_pars['tolerance_grad'], 
                                  tolerance_change=lbfgs_pars['tolerance_change'], 
                                  history_size=lbfgs_pars['history_size'], 
                                  line_search_fn=lbfgs_pars['line_search_fn'])

    roller_outer_LBFGS_full_backsolve.train()
    for i in range(n_steps//n_chunks):

        loss = ((roller_outer_LBFGS_full_backsolve.forward() - target)**2).mean()
        
        if torch.isnan(loss):
            i_ += 1
            continue

        def closure():
            loss = ((roller_outer_LBFGS_full_backsolve.forward() - target)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            return loss            
        optimizer.step(closure)        
        loss_vals_LBFGS_full_backsolve[i_] = loss.detach().cpu().numpy()
        time_vals_LBFGS_full_backsolve[i_] = time.time() - time_vals_LBFGS_full_backsolve[i_]
        print((time_vals_LBFGS_full_backsolve[i_], loss_vals_LBFGS_full_backsolve[i_]))
        i_ += 1
        
    x_sols_LBFGS_full_backsolve[j] = sortL96fromChannels(roller_outer_LBFGS_full_backsolve.X.detach().cpu().numpy().copy())
    grndtrth = out[n_starts+T_rollout-(j+1)*(T_rollout//n_chunks)]
    state_mses_LBFGS_full_backsolve[j] = ((x_sols_LBFGS_full_backsolve[j] - grndtrth)**2).mean()

# ## plot and compare results

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
