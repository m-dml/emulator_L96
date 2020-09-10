import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import L96sim

from L96_emulator.util import dtype, dtype_np, device
from L96_emulator.run_DA import setup_4DVar

res_dir = '/gpfs/work/nonnenma/results/emulators/L96/'
data_dir = '/gpfs/work/nonnenma/data/emulators/L96/'


######################
# load DA experiment #
######################


exp_id = '08'
exp_names = os.listdir('experiments_DA/')   
conf_exp = exp_names[np.where(np.array([name[:2] for name in exp_names])==str(exp_id))[0][0]][:-4]
args = setup_4DVar(conf_exp=f'experiments_DA/{conf_exp}.yml')

save_dir = 'results/data_assimilation/' + args['exp_id'] + '/'
fn = save_dir + 'out.npy'
out = np.load(res_dir + fn, allow_pickle=True)[()]

J, n_steps, T_win, dt = args['J'], args['n_steps'], args['T_win'], args['dt']

# load data: data is ground-truth, y is observed, m is observation mask
data, y, m = out['out'], out['y'], out['m']
x_sols = out['x_sols'] # solutions from 4D-Var optimization


#################
# load emulator #
#################


from L96_emulator.data_assimilation import ObsOp_identity, ObsOp_subsampleGaussian, GenModel, get_model, as_tensor
from L96_emulator.util import sortL96fromChannels, sortL96intoChannels
import torch

K,J = args['K'], args['J']

model_pars = {
    'exp_id' : args['model_exp_id'],
    'model_forwarder' : 'rk4_default',
    'K_net' : args['K'],
    'J_net' : args['J'],
    'dt_net' : args['dt']
}

# ### instantiate model resolvent
model, model_forwarder, _ = get_model(model_pars, res_dir=res_dir, exp_dir='')

# ### instantiate observation operator
ObsOp = ObsOp_subsampleGaussian if args['obs_operator']=='ObsOp_subsampleGaussian' else ObsOp_identity
model_observer = ObsOp(**{'r' : args['obs_operator_r'], 'sigma2' : args['obs_operator_sig2']})

# place-holder prior over initial states
prior = torch.distributions.normal.Normal(loc=torch.zeros((1,J+1,K)), 
                                          scale=1.*torch.ones((1,J+1,K)))

# ### define generative model for observed data
gen = GenModel(model_forwarder, model_observer, prior, T=T_win, x_init=None)


#####################
# define likelihood #
#####################


# using first observation window of length T_win after spin-up phase
n = 0 # pick a trial number
y_in = sortL96intoChannels(as_tensor(y[:T_win,n:n+1]),J=J)
m_in = as_tensor(m[:T_win,n:n+1])

def log_prob(x):
    return gen.log_prob(y=y_in, m=m_in, x=sortL96intoChannels(x,J=J), T_obs=np.arange(T_win))[0]
print('log-prob test (all-zero init.state)', log_prob(torch.zeros(K)).detach().cpu().numpy())




##############
# set up HMC #
##############

import hamiltorch
import matplotlib.pyplot as plt
hamiltorch.set_random_seed(123)

params_init = as_tensor(x_sols[0,n].flatten()) # skip some burnin by initiatilizing from 4D-Var est.
print('log-prob (x_init)', log_prob(params_init).detach().cpu().numpy())

step_size = 0.01   # this has to be small or the emulator might blow up ! Just init, will be refined 
num_samples = 100
burn = 10 

L = 100 # number of leapfrog steps per sample (effectively MCMC thinning?) 
desired_accept_rate = 0.75

params_hmc_nuts = hamiltorch.sample(log_prob_func=log_prob, 
                                    params_init=params_init, 
                                    num_samples=num_samples,
                                    step_size=step_size, 
                                    num_steps_per_sample=L,
                                    desired_accept_rate=desired_accept_rate,
                                    sampler=hamiltorch.Sampler.HMC_NUTS,
                                    burn=burn)

out_nuts = torch.stack(params_hmc_nuts)
with torch.no_grad():
    lls = torch.stack([log_prob(out_nuts[i]) for i in range(out_nuts.shape[0])]) 
print('lls', lls.detach().cpu().numpy())
