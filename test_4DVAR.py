import numpy as np
import torch 

from L96_emulator.util import dtype, dtype_np, device, as_tensor
from L96_emulator.util import sortL96fromChannels, sortL96intoChannels
from L96_emulator.util import predictor_corrector, rk4_default, get_data

from L96_emulator.likelihood import ObsOp_identity, ObsOp_subsampleGaussian, GenModel, SimplePrior

from L96_emulator.data_assimilation import get_model, solve_4dvar


# setup

res_dir = '/gpfs/work/nonnenma/results/emulators/L96/'
data_dir = '/gpfs/work/nonnenma/data/emulators/L96/'

K = 40
J = 0
T = 5
dt = 0.05
spinup_time = 50
N_trials = 5
F = 8.
T_win = 10

model_exp_id = 26
model_forwarder = 'rk4_default'
obs_operator=='ObsOp_subsampleGaussian'
obs_operator_r = 0.5
obs_operator_sig2 = 1.0
B = 1.

model_pars = {
    'exp_id' : model_exp_id,
    'model_forwarder' : model_forwarder,
    'K_net' : K,
    'J_net' : J,
    'dt_net' : dt
}

optimizer_pars = {
              'optimizer' : 'LBFGS',
              'n_steps' : 50,
              'lr' : 1.0,
              'max_iter' : 1000,
              'max_eval' : None,
              'tolerance_grad' : 1e-12,
              'tolerance_change' : 1e-12,
              'history_size': 10
}


# start script

if model_forwarder == 'rk4_default':
    model_forwarder = rk4_default
if model_forwarder == 'predictor_corrector':
    model_forwarder = predictor_corrector

out, datagen_dict = get_data(K=K, J=J, T=T+spinup_time, dt=dt, N_trials=N_trials, 
                             F=F, 
                             resimulate=True, solver=model_forwarder,
                             save_sim=False, data_dir='')
out = sortL96intoChannels(out.transpose(1,0,2)[int(spinup_time/dt):], J=J)
print('out.shape', out.shape)


model, model_forwarder, args = get_model(model_pars, res_dir=res_dir, exp_dir='')

obs_pars = {}
if obs_operator=='ObsOp_subsampleGaussian':
    obs_pars['obs_operator'] = ObsOp_subsampleGaussian
    obs_pars['obs_operator_args'] = {'r' : obs_operator_r, 'sigma2' : obs_operator_sig2}
elif obs_operator=='ObsOp_identity':
    obs_pars['obs_operator'] = ObsOp_identity
    obs_pars['obs_operator_args'] = {}
else:
    raise NotImplementedError()
model_observer = obs_pars['obs_operator'](**obs_pars['obs_operator_args'])


prior = torch.distributions.normal.Normal(loc=torch.zeros((1,J+1,K)), 
                                          scale=B*torch.ones((1,J+1,K)))
gen = GenModel(model_forwarder, model_observer, prior)

y = sortL96fromChannels(gen._sample_obs(as_tensor(out))) # sets the loss masks!
m = torch.stack(gen.masks,dim=0)

print('shapes', (y.shape, m.shape))

print('4D-VAR')
x_sols, losses, times = solve_4dvar(y, m, 
                                    T_obs=np.arange(y.shape[0]), 
                                    T_win=T_win, 
                                    x_init=None, 
                                    model_pars=model_pars, 
                                    obs_pars=obs_pars, 
                                    optimizer_pars=optimizer_pars,
                                    res_dir=res_dir)

np.save(res_dir + 'results/data_assimilation/4D_var_test', 
        {'out' : out,
         'y' : y.detach().cpu().numpy(),
         'm' : m.detach().cpu().numpy(),
         'x_sols' : x_sols,
         'losses' : losses,
         'times' : times,
         'T_win' : T_win
        })
print('x_sols.shape', x_sols.shape)
print('done')
