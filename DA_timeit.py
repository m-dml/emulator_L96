import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import L96sim

from L96_emulator.util import dtype, dtype_np, device

res_dir = '/gpfs/work/nonnenma/results/emulators/L96/'
data_dir = '/gpfs/work/nonnenma/data/emulators/L96/'

from L96_emulator.run import setup

conf_exp = '20_minimalNet_predictState' #'template'
args = setup(conf_exp=f'experiments/{conf_exp}.yml')
args.pop('conf_exp')


from L96sim.L96_base import f1, f2, J1, J1_init, f1_juliadef, f2_juliadef
from L96_emulator.util import predictor_corrector, rk4_default
from L96_emulator.run import sel_dataset_class

try: 
    K, J, T, dt = args['K'], args['J'], args['T'], args['dt']
    spin_up_time, train_frac = args['spin_up_time'], args['train_frac']
    normalize_data = bool(args['normalize_data'])
except:
    K, J, T, dt = 36, 10, 605, 0.001
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

resimulate, save_sim = False, False
if resimulate:
    print('simulating data')
    X_init = F * (0.5 + np.random.randn(K*(J+1)) * 1.0).astype(dtype=dtype_np) / np.maximum(J,50)
    dX_dt = np.empty(X_init.size, dtype=X_init.dtype)
    times = np.linspace(0, T, int(np.floor(T/dt)+1))
    
    out = predictor_corrector(fun=fun, y0=X_init.copy(), times=times, alpha=0.5)

    # filename for data storage
    if save_sim: 
        np.save(data_dir + fn_data, out.astype(dtype=dtype_np))
else:
    print('loading data')
    out = np.load(data_dir + fn_data + '.npy')
    
import torch 
import numpy as np
from L96_emulator.eval import load_model_from_exp_conf

model, model_forwarder, training_outputs = load_model_from_exp_conf(res_dir, args)

from L96_emulator.eval import sortL96fromChannels, sortL96intoChannels

if J > 0:
    def fun(t, x):
        return f2(x, F, h, b, c, dX_dt, K, J)
else:
    def fun(t, x):
        return f1(x, F, dX_dt, K)
dX_dt = np.empty(K*(J+1), dtype=dtype_np)
T_start = np.array([50000, 100000, 15000])
i = 0
for i in range(len(T_start)):
    inputs = out[T_start[i]]
    inputs_torch = torch.as_tensor(sortL96intoChannels(np.atleast_2d(out[T_start[i]]),J=J),dtype=dtype,device=device)

    MSE = ((fun(0., inputs) - sortL96fromChannels(model.forward(inputs_torch).detach().cpu().numpy()))**2).mean()
    print(MSE)
    
prediction_task = 'state'
lead_time = 1
DatasetClass = sel_dataset_class(prediction_task=prediction_task)
dg_train = DatasetClass(data=out, J=J, offset=lead_time, normalize=normalize_data, 
                   start=int(spin_up_time/dt), 
                   end=int(np.floor(out.shape[0]*train_frac)))

from L96_emulator.eval import get_rollout_fun, plot_rollout
from L96_emulator.eval import solve_from_init
from L96_emulator.util import sortL96fromChannels, sortL96intoChannels

model_simulate = get_rollout_fun(dg_train, model_forwarder, prediction_task)

T_start, T_dur = 10*int(spin_up_time/dt), 1000

from L96_emulator.eval import Rollout
roller_outer = Rollout(model_forwarder, prediction_task='state', K=K, J=J, N=1, T=T_dur,
                       x_init=sortL96intoChannels(np.atleast_2d(out[T_start]).copy(),J=J))
roller_outer = torch.jit.script(roller_outer)

target = sortL96intoChannels(torch.as_tensor(out[T_start+T_dur].reshape(1,-1), dtype=dtype, device=device), J=J)