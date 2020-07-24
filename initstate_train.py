import numpy as np
import os
import sys
import L96sim

res_dir = '/gpfs/work/nonnenma/results/emulators/L96/'
data_dir = '/gpfs/work/nonnenma/data/emulators/L96/'

from L96_emulator.util import init_torch_device
import torch

device = init_torch_device()
dtype, dtype_np = torch.float32, np.float32

from L96sim.L96_base import f1, f2, J1, J1_init, f1_juliadef, f2_juliadef
from L96_emulator.util import predictor_corrector
from L96_emulator.run import sel_dataset_class

F, h, b, c = 10, 1, 10, 10
K, J, T, dt = 36, 10, 605, 0.001
spin_up_time, train_frac = 5., 0.8

N, T_rollout = 100, 100
T_start = int(np.ceil(spin_up_time/dt)) * np.arange(1,N+1, dtype=np.int)
print('T_start \n', T_start)

exp_id = 'minimalnet_fullyconn_skipconn_J10'
lead_time = 1
prediction_task = 'state'

save_dir = res_dir + 'models/' + exp_id + '/'
model_fn = f'{exp_id}_dt{lead_time}.pt'
results_fn = f'_rollout_outputs_K{K}_J{J}_T{T}_N{N}_TR{T_rollout}'
output_fn = f'_rollout_training_outputs_K{K}_J{J}_T{T}_N{N}_TR{T_rollout}'

# get data

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
    X_init = F * (0.5 + np.random.randn(K*(J+1)) * 1.0).astype(dtype=dtype_np) / np.maximum(J,10)
    dX_dt = np.empty(X_init.size, dtype=X_init.dtype)
    times = np.linspace(0, T, np.floor(T/dt)+1)

    out = predictor_corrector(fun=fun, y0=X_init.copy(), times=times, alpha=0.5)

    # filename for data storage
    if save_sim: 
        np.save(data_dir + fn_data, out.astype(dtype=dtype_np))
else:
    print('loading data')
    out = np.load(data_dir + fn_data + '.npy')

DatasetClass = sel_dataset_class(prediction_task=prediction_task)
dg_train = DatasetClass(data=out, J=J, offset=lead_time, normalize=False, 
                   start=int(spin_up_time/dt), 
                   end=int(np.floor(out.shape[0]*train_frac)))


# get model

from L96_emulator.networks import MinimalNetL96
from L96_emulator.util import sortL96fromChannels, sortL96intoChannels

model = MinimalNetL96(K,J,F,b,c,h,skip_conn=True,loc=1e3)
std_out = torch.as_tensor(dg_train.std, device=device, dtype=dtype)
mean_out = torch.as_tensor(dg_train.mean, device=device, dtype=dtype)

def model_forward(x):
    alpha = 0.5
    ndim = x.ndim

    x = sortL96fromChannels(x * std_out + mean_out) if ndim == 3 else x

    f0 = model.forward(x)
    f1 = model.forward(x + dt*f0)

    x = x + dt * (alpha*f0 + (1-alpha)*f1)
    x = (sortL96intoChannels(x, J=J) - mean_out) / std_out

    return  sortL96fromChannels(x) if ndim == 2 else x


# optimize initial states

from L96_emulator.eval import Rollout

roller_outer = Rollout(model_forward, prediction_task='state', K=K, J=J, N=N)
target = torch.as_tensor(out[T_start+T_rollout], dtype=dtype, device=device)
x_init = roller_outer.X.detach().cpu().numpy().copy()

n_steps, lr, weight_decay = 100000, 1e-2, 0.
roller_outer.train()
optimizer = torch.optim.Adam(roller_outer.parameters(), lr=lr, weight_decay=weight_decay)
loss_vals = np.zeros(n_steps)
print('starting optimization')
for i in range(n_steps):
    optimizer.zero_grad()
    loss = ((roller_outer.forward(T=T_rollout) - target)**2).mean()
    loss.backward()
    optimizer.step()
    loss_vals[i] = loss.detach().cpu().numpy()

    if np.mod(i, n_steps//10) == 0:
        print(f'-- training done {100*i/n_steps}%')

torch.save(roller_outer.state_dict(), save_dir+results_fn)
np.save(save_dir + output_fn, dict(training_loss=loss_vals, validation_loss=None,
                                   T_start=T_start, T_rollout=T_rollout, x_init=x_init))
print('finished')
