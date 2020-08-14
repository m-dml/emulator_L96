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

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import L96sim

from L96_emulator.util import dtype, dtype_np, device

res_dir = '/gpfs/work/nonnenma/results/emulators/L96/'
data_dir = '/gpfs/work/nonnenma/data/emulators/L96/'


# ### load / simulate data

# In[2]:


from L96sim.L96_base import f1, f2, J1, J1_init, f1_juliadef, f2_juliadef
from L96_emulator.util import predictor_corrector, rk4_default
from L96_emulator.run import sel_dataset_class

try: 
    K, J, T, dt = args['K'], args['J'], args['T'], args['dt']
    spin_up_time, train_frac = args['spin_up_time'], args['train_frac']
    normalize_data = bool(args['normalize_data'])
except:
    K, J, T, dt = 36, 10, 605, 0.01
    spin_up_time, train_frac = 5., 0.8
    normalize_data = False

F, h, b, c = 10, 1, 10, 10

fn_data = f'out_K{K}_J{J}_T{T}_dt0_{str(dt)[2:]}'
if J > 0:
    def fun(t, x):
        return f2(x, F, h, b, c, dX_dt, K, J)
else:
    def fun(t, x):
        return f1(x, F, dX_dt, K)

resimulate, save_sim = True, False
if resimulate:
    print('simulating data')
    X_init = F * (0.5 + np.random.randn(K*(J+1)) * 1.0).astype(dtype=dtype_np) / np.maximum(J,50)
    dX_dt = np.empty(X_init.size, dtype=X_init.dtype)
    times = np.linspace(0, T, int(np.floor(T/dt)+1))
    
    out = rk4_default(fun=fun, y0=X_init.copy(), times=times)

    # filename for data storage
    if save_sim: 
        np.save(data_dir + fn_data, out.astype(dtype=dtype_np))
else:
    print('loading data')
    out = np.load(data_dir + fn_data + '.npy')

"""
plt.figure(figsize=(8,4))
plt.imshow(out.T, aspect='auto')
plt.xlabel('time')
plt.ylabel('location')
plt.show()
"""

prediction_task = 'state'
lead_time = 1
DatasetClass = sel_dataset_class(prediction_task=prediction_task,N_trials=1)
dg_train = DatasetClass(data=out, J=J, offset=lead_time, normalize=normalize_data, 
                   start=int(spin_up_time/dt), 
                   end=int(np.floor(out.shape[0]*train_frac)))


# ### pick a (trained) emulator

# In[3]:


from L96_emulator.run import setup

exp_id = 20

exp_names = os.listdir('experiments/')   
conf_exp = exp_names[np.where(np.array([name[:2] for name in exp_names])==str(exp_id))[0][0]][:-4]

print('conf_exp', conf_exp)

args = setup(conf_exp=f'experiments/{conf_exp}.yml')
args.pop('conf_exp')


# ### choose numerical solver scheme

# In[4]:


args['model_forwarder'] = 'rk4_default'
args['dt_net'] = dt 


# ### load & instantiate the emulator

# In[5]:


import torch 
import numpy as np
from L96_emulator.eval import load_model_from_exp_conf

model, model_forwarder, training_outputs = load_model_from_exp_conf(res_dir, args)

if not training_outputs is None:
    training_loss, validation_loss = training_outputs['training_loss'], training_outputs['validation_loss']

    """
    fig = plt.figure(figsize=(8,8))
    seq_length = args['seq_length']
    plt.semilogy(validation_loss, label=conf_exp+ f' ({seq_length * (J+1)}-dim)')
    plt.title('training')
    plt.ylabel('validation error')
    plt.legend()
    fig.patch.set_facecolor('xkcd:white')
    plt.show()
    """

from L96_emulator.eval import sortL96fromChannels, sortL96intoChannels

if J > 0:
    def fun(t, x):
        return f2(x, F, h, b, c, dX_dt, K, J)
else:
    def fun(t, x):
        return f1(x, F, dX_dt, K)
dX_dt = np.empty(K*(J+1), dtype=dtype_np)
n_starts = np.array([5000, 10000, 15000])
i = 0
for i in range(len(n_starts)):
    inputs = out[n_starts[i]]
    inputs_torch = torch.as_tensor(sortL96intoChannels(np.atleast_2d(out[n_starts[i]]),J=J),dtype=dtype,device=device)

    MSE = ((fun(0., inputs) - sortL96fromChannels(model.forward(inputs_torch).detach().cpu().numpy()))**2).mean()
    print(MSE)


# ### simulate an example rollout from the emulator

# In[6]:


from L96_emulator.eval import get_rollout_fun, plot_rollout
from L96_emulator.util import sortL96fromChannels, sortL96intoChannels

model_simulate = get_rollout_fun(dg_train, model_forwarder, prediction_task)

T_start, T_dur = 10*spin_up_time, 10
n_start, n_dur = int(T_start/dt), int(T_dur/dt)

out_model = model_simulate(y0=dg_train[n_start].copy(), 
                           dy0=dg_train[n_start]-dg_train[n_start-dg_train.offset],
                           n_steps=n_dur)
out_model = sortL96fromChannels(out_model * dg_train.std + dg_train.mean)

solver_comparison = True 
if solver_comparison:
    try: 
        print(F, h, b, c)
    except: 
        F, h, b, c = 10, 1, 10, 10
    1
    times_ = np.linspace(0, T_dur, 2*n_dur+1) # + n_start
    out2 = rk4_default(fun=fun, y0=out[n_start], times=times_)[::2]
else:
    out2 = None

#fig = plot_rollout(out, out_model, out_comparison=out2, n_start=n_start, n_steps=n_dur, K=K)


# # Solving a fully-observed inverse problem

# In[7]:


n_starts = np.arange(int(spin_up_time/dt), int(train_frac*out.shape[0]), 2* int(spin_up_time/dt))
T_rollout, N = 40, len(n_starts)
n_chunks = 10


# In[ ]:


"""
import time
from L96_emulator.eval import Rollout

if J > 0:
    def negfun(t, x):
        return - f2(x, F, h, b, c, dX_dt, K, J)
else:
    def negfun(t, x):
        return - f1(x, F, dX_dt, K)

n_steps, lr, weight_decay = 200, 1.0, 0.0

loss_vals_backsolve = np.zeros(n_steps)
time_vals_backsolve = time.time() * np.ones(n_steps)

target = out[n_starts+T_rollout]
state_mses_backsolve = np.zeros(n_chunks)

x_init = np.zeros((len(n_starts), K*(J+1)))

i_ = 0

plt.figure(figsize=(12, 4))
for c_, dt_fac in enumerate([1, 10, 100, 1000]):
    plt.subplot(1,4,c_+1)
    for j in range(n_chunks):

        T_i = (j+1)*T_rollout//n_chunks
        times = np.linspace(0, dt*T_i, dt_fac*T_i+1)
        print('backward solving')
        for i__ in range(len(n_starts)):
            out2 = rk4_default(fun=negfun, y0=out[n_starts[i__]+T_rollout].copy(), times=times)
            x_init[i__] = out2[-1].copy()
        state_mses_backsolve[j] = ((x_init - target)**2).mean()
    plt.plot(T_rollout//n_chunks*np.arange(1,n_chunks+1), state_mses_backsolve)
    plt.xlabel('T_rollout')
    plt.ylabel('MSE of iniitial state estimate')
    plt.title(f'dt={dt/dt_fac}')
plt.show()
"""


# ## L-BFGS, split rollout time into chunks, solve sequentially from end to beginning

# In[ ]:


import time
from L96_emulator.eval import Rollout

n_steps, lr, weight_decay = 1000, 1.0, 0.0

loss_vals_LBFGS_chunks = np.zeros(n_steps)
time_vals_LBFGS_chunks = time.time() * np.ones(n_steps)
loss_vals_LBFGS_chunks_rollout = np.zeros_like(loss_vals_LBFGS_chunks)

x_sols_LBFGS_chunks = np.zeros((n_chunks, N, K*(J+1)))
state_mses_LBFGS_chunks = np.zeros(n_chunks)

grndtrth_rollout = sortL96intoChannels(torch.as_tensor(out[n_starts], dtype=dtype, device=device), J=J)
target_rollout = sortL96intoChannels(torch.as_tensor(out[n_starts+T_rollout], dtype=dtype, device=device),J=J)
T_rollout_i = (T_rollout//n_chunks) * np.ones(n_chunks, dtype=np.int)

x_inits = np.zeros((n_chunks, N, K*(J+1)))
x_init = sortL96intoChannels(np.atleast_2d(out[n_starts+T_rollout].copy()),J=J)
targets = np.zeros((n_chunks, N, K*(J+1)))
targets[0] = out[n_starts+T_rollout]

i_ = 0
for j in range(n_chunks):
    roller_outer_LBFGS_chunks = Rollout(model_forwarder, prediction_task='state', K=K, J=J, 
                                        N=N, T=T_rollout_i[j], 
                                        x_init=x_init)
    x_inits[j] = sortL96fromChannels(roller_outer_LBFGS_chunks.X.detach().cpu().numpy().copy())
    optimizer = torch.optim.LBFGS(params=roller_outer_LBFGS_chunks.parameters(), 
                                  lr=lr, 
                                  max_iter=100, 
                                  max_eval=None, 
                                  tolerance_grad=1e-07, 
                                  tolerance_change=1e-09, 
                                  history_size=50, 
                                  line_search_fn='strong_wolfe')
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
        roller_outer_LBFGS_chunks.T = T_rollout_i[j]

        i_ += 1

    x_init = roller_outer_LBFGS_chunks.X.detach().cpu().numpy().copy()
    if j < n_chunks - 1:
        targets[j+1] = sortL96fromChannels(roller_outer_LBFGS_chunks.X.detach().cpu().numpy().copy())

    x_sols_LBFGS_chunks[j] = sortL96fromChannels(roller_outer_LBFGS_chunks.X.detach().cpu().numpy().copy())
    grndtrth = out[n_starts+T_rollout-(j+1)*(T_rollout//n_chunks)]
    state_mses_LBFGS_chunks[j] = ((x_sols_LBFGS_chunks[j] - grndtrth)**2).mean()

    
"""
plt.figure(figsize=(8,2))
plt.semilogy(loss_vals_LBFGS_chunks, label='initialization')
plt.title('rollout chunk state loss across gradient descent steps')
plt.ylabel('MSE)')
plt.xlabel('gradient step')
plt.show()

plt.figure(figsize=(8,2))
plt.semilogy(loss_vals_LBFGS_chunks_rollout, label='initialization')
plt.title('rollout final state loss across gradient descent steps')
plt.ylabel('MSE)')
plt.xlabel('gradient step')
plt.show()
"""


# ## L-BFGS, solve across full rollout time in one go
# - warning, this can be excruciatingly slow and hard to converge !

# In[ ]:



import time
from L96_emulator.eval import Rollout

n_steps, lr, weight_decay = 1000, 1.0, 0.0

loss_vals_LBFGS_full_persistence = np.zeros(n_steps)
time_vals_LBFGS_full_persistence = time.time() * np.ones(n_steps)

x_sols_LBFGS_full_persistence = np.zeros((n_chunks, N, K*(J+1)))
state_mses_LBFGS_full_persistence = np.zeros(n_chunks)

x_init = sortL96intoChannels(out[n_starts+T_rollout].copy(), J=J)
target = sortL96intoChannels(torch.as_tensor(out[n_starts+T_rollout], dtype=dtype, device=device), J=J)

i_ = 0
for j in range(n_chunks):

    roller_outer_LBFGS_full = Rollout(model_forwarder, prediction_task='state', K=K, J=J, 
                                        N=N, T=(j+1)*T_rollout//n_chunks, x_init=x_init)
    optimizer = torch.optim.LBFGS(params=roller_outer_LBFGS_full.parameters(),
                                  lr=lr,
                                  max_iter=100,
                                  max_eval=None,
                                  tolerance_grad=1e-07, 
                                  tolerance_change=1e-09,
                                  history_size=50,
                                  line_search_fn='strong_wolfe')
    roller_outer_LBFGS_full.train()
    for i in range(n_steps//n_chunks):

        loss = ((roller_outer_LBFGS_full.forward() - target)**2).mean()
        
        if torch.isnan(loss):
            i_ += 1
            continue

        def closure():
            loss = ((roller_outer_LBFGS_full.forward() - target)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            return loss            
        optimizer.step(closure)        
        loss_vals_LBFGS_full_persistence[i_] = loss.detach().cpu().numpy()
        time_vals_LBFGS_full_persistence[i_] = time.time() - time_vals_LBFGS_full_persistence[i_]
        print((time_vals_LBFGS_full_persistence[i_], loss_vals_LBFGS_full_persistence[i_]))
        i_ += 1

    x_sols_LBFGS_full_persistence[j] = sortL96fromChannels(roller_outer_LBFGS_chunks.X.detach().cpu().numpy().copy())
    grndtrth = out[n_starts+T_rollout-(j+1)*(T_rollout//n_chunks)]
    state_mses_LBFGS_full_persistence[j] = ((x_sols_LBFGS_full_persistence[j] - grndtrth)**2).mean()

"""
plt.figure(figsize=(8,2))
plt.semilogy(loss_vals_LBFGS_full_persistence, label='initialization')
plt.title('rollout final state loss across gradient descent steps')
plt.ylabel('MSE)')
plt.xlabel('gradient step')
plt.show()
"""


# ## L-BFGS, solve across full rollout time in one go, initialize from chunked approach

# In[ ]:


import time
from L96_emulator.eval import Rollout

n_steps, lr, weight_decay = 1000, 1.0, 0.0

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
                                  lr=lr, 
                                  max_iter=100, 
                                  max_eval=None, 
                                  tolerance_grad=1e-07, 
                                  tolerance_change=1e-09, 
                                  history_size=50, 
                                  line_search_fn='strong_wolfe')
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

    x_sols_LBFGS_full_chunks[j] = sortL96fromChannels(roller_outer_LBFGS_chunks.X.detach().cpu().numpy().copy())
    grndtrth = out[n_starts+T_rollout-(j+1)*(T_rollout//n_chunks)]
    state_mses_LBFGS_full_chunks[j] = ((x_sols_LBFGS_full_chunks[j] - grndtrth)**2).mean()

"""
plt.figure(figsize=(8,2))
plt.semilogy(loss_vals_LBFGS_full_chunks, label='initialization')
plt.title('rollout final state loss across gradient descent steps')
plt.ylabel('MSE)')
plt.xlabel('gradient step')
plt.show()
"""


# ## L-BFGS, solve across full rollout time in one go, initiate from backward solution

# In[9]:


import time
from L96_emulator.eval import Rollout

if J > 0:
    def fun(t, x):
        return - f2(x, F, h, b, c, dX_dt, K, J)
else:
    def fun(t, x):
        return - f1(x, F, dX_dt, K)

n_steps, lr, weight_decay = 1000, 1.0, 0.0

loss_vals_LBFGS_full_backsolve = np.zeros(n_steps)
time_vals_LBFGS_full_backsolve = time.time() * np.ones(n_steps)

x_sols_backsolve = np.zeros((n_chunks, N, K*(J+1)))
x_sols_LBFGS_full_backsolve = np.zeros((n_chunks, N, K*(J+1)))
state_mses_LBFGS_full_backsolve = np.zeros(n_chunks)

target = sortL96intoChannels(torch.as_tensor(out[n_starts+T_rollout], dtype=dtype, device=device), J=J)
x_init = np.zeros((len(n_starts), K*(J+1)))

state_mses_backsolve = np.zeros(n_chunks)
time_vals_backsolve = np.zeros(n_steps)

i_ = 0
for j in range(n_chunks):
    
    T_i = (j+1)*T_rollout//n_chunks
    times = dt * np.linspace(0, T_i, 100 * T_i+1) # note the 100x increase in temporal resolution!
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
                                  lr=lr, 
                                  max_iter=100, 
                                  max_eval=None, 
                                  tolerance_grad=1e-07, 
                                  tolerance_change=1e-09, 
                                  history_size=50, 
                                  line_search_fn='strong_wolfe')

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

"""
plt.figure(figsize=(8,2))
plt.semilogy(loss_vals_LBFGS_full_backsolve, label='initialization')
plt.title('rollout final state loss across gradient descent steps')
plt.ylabel('MSE)')
plt.xlabel('gradient step')
plt.show()
"""


# ## plot and compare results

# In[11]:


np.save(res_dir + 'results/data_assimilation/fullyobs_initstate_tests',
        arr={
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
"""
res = np.load(res_dir + 'results/data_assimilation/fullyobs_initstate_tests.npy', allow_pickle=True)[()]

loss_vals_LBFGS_full_backsolve=res['loss_vals_LBFGS_full_backsolve']
loss_vals_LBFGS_full_persistence=res['loss_vals_LBFGS_full_persistence']
loss_vals_LBFGS_full_chunks=res['loss_vals_LBFGS_full_chunks']
loss_vals_LBFGS_chunks_rollout=res['loss_vals_LBFGS_chunks_rollout']
time_vals_LBFGS_full_backsolve=res['time_vals_LBFGS_full_backsolve']
time_vals_LBFGS_full_persistence=res['time_vals_LBFGS_full_persistence']
time_vals_LBFGS_full_chunks=res['time_vals_LBFGS_full_chunks']
time_vals_LBFGS_chunks=res['time_vals_LBFGS_chunks']
"""


# In[22]:

"""
appr_names = ['full optim, init from backsolve', 'full optim, init from chunks',
              'full optim, init from persistence',
              'optim over single chunk (full rollout error)']
all_losses = [loss_vals_LBFGS_full_backsolve, loss_vals_LBFGS_full_chunks, loss_vals_LBFGS_full_persistence, loss_vals_LBFGS_chunks_rollout]
all_times =  [time_vals_LBFGS_full_backsolve, time_vals_LBFGS_full_chunks, time_vals_LBFGS_full_persistence, time_vals_LBFGS_chunks]
all_losses, all_times

plt.figure(figsize=(16,8))
for i,loss in enumerate(all_losses):
    xx = np.arange(len(loss))+1 if len(loss) == 1000 else np.arange(0, 10*len(loss), 10)+1
    plt.semilogy(xx, loss, label=appr_names[i])        

try:
    loss = loss_vals_LBFGS_chunks
    xx = np.arange(len(loss))+1 if len(loss) == 1000 else np.arange(0, 10*len(loss), 10)+1
    plt.semilogy(xx, loss, 'k--', alpha=0.3, label='optim over single chunk (current chunk error)')        
except:
    pass

for i in range(n_chunks):
    plt.semilogy(n_steps*i + n_steps*np.array([0.05, 0.95]), 1e-10*np.ones(2), 'k')
    plt.text(i*(n_steps), 5e-11, f'T_rollout={(i+1)*T_rollout//n_chunks}')

plt.legend()
plt.xlabel('# gradient step')
plt.ylabel('rollout MSE')
plt.suptitle('optimization error for different initialization methods')
plt.show()
"""

# ### compare with plain gradient descent (SGD with single data point)

# In[ ]:


"""

import time
from L96_emulator.eval import Rollout


n_starts = np.array([5000, 10000, 15000])
T_rollout, N = 100, len(n_starts)
n_chunks = 20

target = torch.as_tensor(out[n_starts+T_rollout], dtype=dtype, device=device)

x_init = out[n_starts+T_rollout].copy()
roller_outer_SGD = Rollout(model_forward, prediction_task='state', K=K, J=J, N=N, x_init=x_init)
x_init = roller_outer_SGD.X.detach().cpu().numpy().copy()

n_steps, lr, weight_decay = 500, 0.01, 0.0
roller_outer_SGD.train()

optimizer = torch.optim.Adam(roller_outer_SGD.parameters(), lr=lr, weight_decay=weight_decay)

#optimizer = torch.optim.LBFGS(params=roller_outer.parameters(), 
#                              lr=lr, 
#                              max_iter=20, 
#                              max_eval=None, 
#                              tolerance_grad=1e-07, 
#                              tolerance_change=1e-09, 
#                              history_size=100, 
#                              line_search_fn=None)
loss_vals_SGD = np.zeros(n_steps)
time_vals_SGD = time.time() * np.ones(n_steps)
for i in range(n_steps):
        optimizer.zero_grad()
        loss = ((roller_outer_SGD.forward(T=T_rollout) - target)**2).mean()
        loss.backward()
        optimizer.step()
        loss_vals_SGD[i] = loss.detach().cpu().numpy()
        time_vals_SGD[i] = time.time() - time_vals_SGD[i]
        print((time_vals_SGD[i], loss_vals_SGD[i]))
        
plt.figure(figsize=(8,2))
plt.semilogy(loss_vals_SGD, label='initialization')
plt.title('rollout final state loss across gradient descent steps')
plt.ylabel('MSE)')
plt.xlabel('gradient step')
plt.show()

"""


# In[ ]:


"""
MSEs_chunks = np.zeros(n_chunks)
MSEs_direct__init_chunks = np.zeros(n_chunks)
MSEs_direct__init_prev = np.zeros(n_chunks)

target = torch.as_tensor(out[n_starts+T_rollout], dtype=dtype, device=device)
for j in range(n_chunks):

    roller_outer_LBFGS_chunks = Rollout(model_forward, prediction_task='state', K=K, J=J, N=N, x_init=x_sols[j])
    MSEs_chunks[j] = ((roller_outer_LBFGS_chunks.forward(T=(j+1)*T_rollout//n_chunks) - target)**2).mean().detach().cpu().numpy()

    #roller_outer_LBFGS_chunks = Rollout(model_forward, prediction_task='state', K=K, J=J, N=N, x_init=x_sols[j])
    #MSEs_chunks[j] = ((roller_outer_LBFGS_chunks.forward(T=(j+1)*T_rollout//n_chunks) - target)**2).mean().detach().cpu().numpy()
"""


# In[ ]:


"""
plt.figure(figsize=(16,16))
for i in range(N):
    plt.subplot(2,N,i+1)
    plt.plot(roller_outer_ADAM.X.detach().cpu().numpy().copy()[i], label='one go')
    plt.plot(roller_outer_test.X.detach().cpu().numpy().copy()[i], '--', label='in 10 chunks')
    plt.plot(roller_outer_LBFGS_chunks.X.detach().cpu().numpy().copy()[i], label='in 10 chunks, L-BFGS')
    plt.legend()

    plt.subplot(2,N,N+i+1)
    plt.plot(roller_outer_ADAM.forward(T=T_rollout).detach().cpu().numpy().copy()[i], label='one go')
    plt.plot(roller_outer_test.forward(T=T_rollout).detach().cpu().numpy().copy()[i], '--', label='in 10 chunks')
    plt.plot(roller_outer_LBFGS_chunks.forward(T=T_rollout).detach().cpu().numpy().copy()[i], '--', label='in 10 chunks, L-BFGS')
    plt.legend()
    
plt.show()
"""


# In[ ]:





# # share notebook results via html file

# In[ ]:


#get_ipython().system("jupyter nbconvert --output-dir='/gpfs/home/nonnenma/projects/lab_coord/mdml_wiki/marcel/emulators' --to html data_assimilation.ipynb")


# In[ ]:



