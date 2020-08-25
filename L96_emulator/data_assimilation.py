import numpy as np
import torch 

import os
import time

from L96_emulator.util import dtype, dtype_np, device
from L96_emulator.util import sortL96fromChannels, sortL96intoChannels
from L96_emulator.util import predictor_corrector, rk4_default, get_data

from L96_emulator.run import setup, sel_dataset_class

from L96_emulator.eval import load_model_from_exp_conf, get_rollout_fun, named_network, Rollout

from L96_emulator.networks import Model_forwarder_predictorCorrector, Model_forwarder_rk4default

import L96sim
from L96sim.L96_base import f1, f2, pf2

def as_tensor(x):
    return torch.as_tensor(x, dtype=dtype, device=device)

class ObsOp_identity(torch.nn.Module):
    def __init__(self):
        super(ObsOp_identity, self).__init__()
        self.sigma = torch.tensor([1.0]) # note we have non-zero sigma for log_prob !
        self.ndistr = torch.distributions.normal.Normal(loc=0., scale=self.sigma)
        self.mask = 1.

    def _sample_mask(self, sample_shape):
        self.mask = torch.ones(size=sample_shape, dtype=torch.int)

    def forward(self, x): # deterministic part of observation operator
        """ y = f(x)  """
        assert len(x.shape) == 3 # N x J+1 x K
        return x

    def sample(self, x, m=None): # observation operator (incl. stochastic parts)
        """ sample y ~ p(y |x, m) """
        if m is None:
            self._sample_mask(sample_shape=x.shape)
            return self.forward(x)
        assert x.shape == m.shape
        return m * self.forward(x)

    def log_prob(self, y, x, m=None):
        """ log p(y|x, m)  """
        assert len(y.shape) == 3 # N x J+1 x K
        if m is None:
            return self.ndistr.log_prob(x - y).sum() # sum from iid over dims
        assert y.shape == m.shape and x.shape == y.shape
        return (m * self.ndistr.log_prob(x - y)).sum(axis=(-2,-1)) # sum from iid over dims


class ObsOp_subsampleGaussian(ObsOp_identity):
    def __init__(self, r=0., sigma2=0.):
        super(ObsOp_subsampleGaussian, self).__init__()
        assert sigma2 >= 0.
        self.sigma2 = as_tensor(sigma2)
        self.sigma = torch.sqrt(self.sigma2)
        self.ndistr = torch.distributions.normal.Normal(loc=0., scale=self.sigma)

        assert 0. <= r <=1.
        self.r = as_tensor(r)
        self.mdistr = torch.distributions.Bernoulli(probs=1-self.r)
        self.mask = 1.

    def _sample_mask(self, sample_shape):
        self.mask = self.mdistr.sample(sample_shape=sample_shape)

    def sample(self, x, m=None): # observation operator (incl. stochastic parts)
        """ sample y ~ p(y |x, m) """
        if m is None:
            self._sample_mask(sample_shape=x.shape)
            m = self.mask
        eps = self.sigma * self.ndistr.sample(sample_shape=x.shape)
        return m * (self.forward(x) + eps)

    def log_prob(self, y, x, m=None):
        """ log p(y|x, m)  """
        assert len(y.shape) == 3 # N x J+1 x K

        x = x.reshape(1, *x.shape) if len(x.shape)==2 else x
        if m is None:
            m = self.mask
        m = m.reshape(1, *m.shape) if len(m.shape)==2 else x
        assert y.shape[1:] == m.shape[1:] and x.shape[1:] == y.shape[1:]

        return (m * self.ndistr.log_prob(x - y)).sum(axis=(-2,-1)) # sum from iid over dims


class GenModel(torch.nn.Module):

    def __init__(self, model_forwarder, model_observer, prior, 
                 T=1, x_init=None):

        super(GenModel, self).__init__()

        self.model_forwarder = model_forwarder
        self.set_rollout_len(T)

        self.model_observer = model_observer
        self.masks = [self.model_observer.mask]

        self.prior = prior        

        # variable container for e.g. maximim-likelihood estimate: 
        x_init = self.prior.sample() if x_init is None else x_init
        assert len(x_init.shape) in [2,3]
        self.set_state(x_init)


    def _forward(self, x=None, T_obs=None):

        x = self.X if x is None else x
        if len(x.shape) == 2:
            x = x.reshape(1, *x.shape)
        assert len(x.shape) == 3 # N x J+1 x K

        T_obs = [self.T-1] if T_obs is None else T_obs

        y = []
        for t in range(np.max(T_obs) + 1):
            x = self.model_forwarder.forward(x)
            if t in T_obs:
                y.append(x)

        return y # returns a list of len(T_obs)!

    def forward(self, x=None, T_obs=None):

        y = [self.model_observer.forward(z) for z in self._forward(x, T_obs)]

        return torch.stack(y, dim=0) # len(T_obs) x N x J+1 x K

    def _sample_obs(self, y, m=None):

        m = [None for t in range(len(y))] if m is None else m
        assert len(m) == len(y)
        self.masks = []
        yn = []
        for i in range(len(y)):
            yn.append(self.model_observer.sample(y[i], m=m[i]))
            self.masks.append(self.model_observer.mask)

        return torch.stack(yn, dim=0) # len(T_obs) x N x J+1 x K

    def sample(self, x=None, m=None, T_obs=None):

        y = self._forward(x, T_obs)
        
        return self._sample_obs(self._forward(x, T_obs), m=m)

    def log_prob(self, y, x=None, m=None, T_obs=None):

        m = self.masks if m is None else m
        xs = self.forward(x, T_obs) # # len(T_obs) x N x J+1 x K

        log_probs = torch.stack([self.model_observer.log_prob(y_, x_, m_) for y_,x_, m_ in zip(y, xs, m)]) # len(T_obs) x N

        return log_probs.sum(axis=0)

    def set_state(self, x_init):

        self.X = torch.nn.Parameter(as_tensor(x_init))

    def set_rollout_len(self, T):

        assert T >= 0
        self.T = T


def mse_loss_fullyObs(x, t):

    assert x.shape == t.shape
    return ((x - t)**2).mean()


def mse_loss_masked(x, t, m):

    assert x.shape == m.shape and t.shape == m.shape
    return (m * ((x - t)**2)).sum() / m.sum()


def optim_initial_state(
      gen,
      T_rollouts, T_obs, N, n_chunks,
      optimizer_pars,
      x_inits, targets, grndtrths, 
      loss_masks=None, f_init=None):

    sample_shape = gen.prior.sample().shape # (..., J+1, K)
    J, K = sample_shape[-2]-1, sample_shape[-1]
    n_steps = optimizer_pars['n_steps']

    x_sols = np.zeros((n_chunks, N, K*(J+1)))
    loss_vals = np.inf * np.ones((n_steps*n_chunks,N))
    time_vals = time.time() * np.ones((n_steps*n_chunks,N))
    state_mses = np.inf * np.ones((n_chunks, N))
    
    loss_masks = [torch.ones((N,J+1,K)) for i in range(n_chunks)] if loss_masks is None else loss_masks
    assert len(loss_masks) == n_chunks

    i_ = 0
    for j in range(n_chunks):

        print('\n')
        print(f'optimizing over chunk #{j+1} out of {n_chunks}')
        print('\n')

        target = sortL96intoChannels(as_tensor(targets[j]),J=J)
        loss_mask = loss_masks[j]

        for n in range(N):

            print('\n')
            print(f'optimizing over initial state #{n+1} / {N}')
            print('\n')


            gen.set_state(x_inits[j][n:n+1])
            gen.set_rollout_len(T_rollouts[j])

            optimizer = torch.optim.LBFGS(params=[gen.X],
                                          lr=optimizer_pars['lr'],
                                          max_iter=optimizer_pars['max_iter'],
                                          max_eval=optimizer_pars['max_eval'],
                                          tolerance_grad=optimizer_pars['tolerance_grad'],
                                          tolerance_change=optimizer_pars['tolerance_change'],
                                          history_size=optimizer_pars['history_size'],
                                          line_search_fn='strong_wolfe')

            for i_n in range(n_steps):

                with torch.no_grad():
                    loss = - gen.log_prob(y=target[:len(T_obs[j]),n:n+1], m=loss_mask[:len(T_obs[j]),n:n+1], T_obs=T_obs[j])
                    if torch.isnan(loss):
                        loss_vals[i_n,n] = loss.detach().cpu().numpy()
                        time_vals[i_+i_n,n] = time.time() - time_vals[i_+i_n,n]
                        continue

                def closure():
                    loss = - gen.log_prob(y=target[:len(T_obs[j]),n:n+1], m=loss_mask[:len(T_obs[j]),n:n+1], T_obs=T_obs[j])
                    optimizer.zero_grad()
                    loss.backward()
                    return loss
                optimizer.step(closure)
                loss_vals[i_+i_n,n] = loss.detach().cpu().numpy()
                time_vals[i_+i_n,n] = time.time() - time_vals[i_+i_n,n]
                print((time_vals[i_n,n], loss_vals[i_n,n]))
            

            x_sols[j][n] = sortL96fromChannels(gen.X.detach().cpu().numpy().copy())
            state_mses[j][n] = ((x_sols[j][n] - grndtrths[j][n])**2).mean()

        # if solving recursively, define next target as current initial state estimate 
        if j < n_chunks - 1 and targets[j+1] is None:
            targets[j+1] = x_sols[j].copy()

        i_ += n_steps

        with torch.no_grad():  
            print('Eucl. distance to initial value', mse_loss_fullyObs(x_sols[j], grndtrths[j]))
            print('Eucl. distance to x_init', mse_loss_fullyObs(x_sols[j], sortL96fromChannels(x_inits[j])))
            try:
                if loss_masks[j] is None:
                    print('Eucl. distance to target', mse_loss_fullyObs(x_sols[j], targets[j]))
                else:
                    print('Eucl. distance to target', mse_loss_masked(x_sols[j], 
                                                                targets[j],
                                                                sortL96fromChannels(loss_masks[j]).detach().cpu().numpy()))
            except:
                pass

        if j < n_chunks - 1 and x_inits[j+1] is None:
            x_inits[j+1] = sortL96intoChannels(x_sols[j], J=J).copy()
            if not f_init is None:
                x_inits[j+1] = f_init(x_inits[j+1])

    # correting time stamps for solving multiple trials sequentially
    for j in range(n_chunks):
        top_new = time_vals[(j+1)*n_steps-1,N-1]
        for n in range(1,N)[::-1]:
            # correct for the fact that n-1 other problems were solve before for this j
            time_vals[j*n_steps+np.arange(n_steps),n] -= time_vals[(j+1)*n_steps-1,n-1]
            if j > 0:
                # continue from j-1 for this n
                time_vals[j*n_steps+np.arange(n_steps),n] += time_vals[j*n_steps-1,n]
        if j > 0:
            # for first trial (n=0) of this j, clear previous time and continue from j-1
            time_vals[j*n_steps+np.arange(n_steps),0] -= top_old
            time_vals[j*n_steps+np.arange(n_steps),0] += time_vals[j*n_steps-1,0]
        top_old = top_new             
                
    return x_sols, loss_vals, time_vals, state_mses


def solve_initstate(system_pars, model_pars, optimizer_pars, setup_pars, optimiziation_schemes, res_dir, data_dir):

    # extract key variable names from input dicts
    K, J = system_pars['K'], system_pars['J']
    T, dt, N_trials = system_pars['T'], system_pars['dt'], system_pars['N_trials']
    
    n_starts, T_rollout = setup_pars['n_starts'], setup_pars['T_rollout']
    n_chunks, n_chunks_recursive = setup_pars['n_chunks'], setup_pars['n_chunks_recursive']

    N = len(n_starts)
    recursions_per_chunks = n_chunks_recursive//n_chunks

    assert recursions_per_chunks == n_chunks_recursive/n_chunks

    assert T_rollout//n_chunks_recursive == T_rollout/n_chunks_recursive
    assert T_rollout//n_chunks == T_rollout/n_chunks
    
    if optimiziation_schemes['LBFGS_full_chunks']:
        assert optimiziation_schemes['LBFGS_chunks'] # requirement for init
        
    if optimiziation_schemes['LBFGS_full_backsolve']:
        assert optimiziation_schemes['backsolve'] # requirement for init
        

    # get model
    model, model_forwarder, args = get_model(model_pars, res_dir=res_dir, exp_dir='')
    
    # ### instantiate observation operator
    model_observer = system_pars['obs_operator'](**system_pars['obs_operator_args'])

    
    # ### define prior over initial states
    prior = torch.distributions.normal.Normal(loc=torch.zeros((1,J+1,K)), 
                                              scale=1.*torch.ones((1,J+1,K)))

    
    # ### define generative model for observed data
    gen = GenModel(model_forwarder, model_observer, prior, T=T_rollout, x_init=None)
    
    # prepare function output
    model_forwarder_str, optimizer_str = args['model_forwarder'], optimizer_pars['optimizer']
    exp_id, obs_operator_str = model_pars['exp_id'], model_observer.__class__.__name__
    fn = 'results/data_assimilation/fullyobs_initstate_tests_'
    fn = fn + f'exp{exp_id}_{model_forwarder_str}_{optimizer_str}_{obs_operator_str}'

    # output dictionary
    res = { 'exp_id' : exp_id,
            'K' : K,
            'J' : J,
            'T' : T, 
            'dt' : dt,

            'spin_up_time' :  system_pars['spin_up_time'],
            'prediction_task' : setup_pars['prediction_task'],
            'train_frac' :  system_pars['train_frac'],
            'normalize_data' : system_pars['normalize_data'],
            'back_solve_dt_fac' : system_pars['back_solve_dt_fac'],
            'F' : system_pars['F'], 
            'h' : system_pars['h'], 
            'b' : system_pars['b'], 
            'c' : system_pars['c'],
            'lead_time' : setup_pars['lead_time'],

            'conf_exp' : args['conf_exp'],
            'model_forwarder' : model_pars['model_forwarder'], # should still be string
            'dt_net' : model_pars['dt_net'],

            'n_starts' : n_starts,
            'T_rollout' : T_rollout,
            'n_chunks' : n_chunks,
            'n_chunks_recursive' : n_chunks_recursive,
            'recursions_per_chunks' : recursions_per_chunks,
            'n_steps' : optimizer_pars['n_steps'],
            'n_steps_tot' : optimizer_pars['n_steps']*n_chunks,

            'optimizer_pars' : optimizer_pars, 
            'optimiziation_schemes' : optimiziation_schemes,

            'obs_operator' : obs_operator_str,
            'obs_operator_args' : system_pars['obs_operator_args']
    }


    # functions for explicitly solving backwards
    dX_dt = np.empty(K*(J+1), dtype=dtype_np)
    if J > 0:
        def fun_eb(t, x):
            return - f2(x, res['F'], res['h'], res['b'], res['c'], dX_dt, K, J)
    else:
        def fun_eb(t, x):
            return - f1(x, res['F'], dX_dt, K)

    def explicit_backsolve(x_init, times_eb, fun_eb):
        x_sols = np.zeros_like(x_init)
        for i__ in range(x_init.shape[0]):
            out2 = rk4_default(fun=fun_eb, y0=x_init[i__], times=times_eb)
            x_sols[i__] = out2[-1].copy()#.detach().cpu().numpy().copy()
        return x_sols
        
    class Model_eb(torch.nn.Module):
        def __init__(self, model):
            super(Model_eb, self).__init__()            
            self.model = model
        def forward(self, x):
            return - self.model.forward(x)

    if res['model_forwarder'] == 'rk4_default':
        Model_forwarder = Model_forwarder_rk4default  
    elif res['model_forwarder'] == 'predictor_corrector':
        Model_forwarder = Model_forwarder_predictorCorrector

    model_forwarder_eb = Model_forwarder(model=Model_eb(model), dt=dt/res['back_solve_dt_fac'])


    # ### get data for 'typical' L96 state sequences

    out, datagen_dict = get_data(K=K, J=J, T=T, dt=dt, N_trials=N_trials, 
                                 F=res['F'], h=res['h'], b=res['b'], c=res['c'], 
                                 resimulate=True, solver=rk4_default,
                                 save_sim=False, data_dir=data_dir)

    grndtrths = [out[n_starts] for j in range(n_chunks_recursive)]
    res['initial_states'] = np.stack([sortL96intoChannels(z,J=J) for z in grndtrths])

    T_obs = [[(j+1)*(T_rollout//n_chunks)-1 for j in range(n_+1)] for n_ in range(n_chunks)]
    print('T_obs[-1]', T_obs[-1])
    res['targets'] = np.stack([sortL96intoChannels(out[n_starts+t+1], J=J) for t in T_obs[-1]], axis=0)
    
    DatasetClass = sel_dataset_class(prediction_task=res['prediction_task'],N_trials=1)
    dg_train = DatasetClass(data=out, J=J, offset=res['lead_time'], normalize=res['normalize_data'], 
                       start=int(res['spin_up_time']/dt), 
                       end=int(np.floor(out.shape[0]*res['train_frac'])))

    # ## Generate observed data: (sub-)sample noisy observations
    res['targets_obs'] = gen._sample_obs(as_tensor(res['targets'])) # sets the loss masks!
    res['targets_obs'] = sortL96fromChannels(res['targets_obs'].detach().cpu().numpy())
    res['loss_mask'] = torch.stack(gen.masks,dim=0).detach().cpu().numpy()

    print('mask . shape', res['loss_mask'].shape)
    print('target . shape', res['targets'].shape)
    print('targets_obs . shape', res['targets_obs'].shape)
    
    print('\n')
    print('storing intermediate results')
    print('\n')
    np.save(res_dir + fn, arr=res)


    # ### define setup for optimization

    T_rollouts = np.arange(1, n_chunks+1) * (T_rollout//n_chunks)
    grndtrths = [out[n_starts] for j in range(n_chunks)]
    targets = [1.*res['targets_obs'][:n_chunks] for i in range(n_chunks)]
    loss_masks = [torch.stack(gen.masks[:n_chunks],dim=0) for i in range(n_chunks)]
    
    T_rollouts_chunks = np.arange(1, n_chunks_recursive+1) * (T_rollout//n_chunks_recursive)
    grndtrths_chunks = [out[n_starts] for j in range(n_chunks_recursive)]


    # ## L-BFGS, solve across full rollout time recursively, initialize from forward solver in reverse

    if optimiziation_schemes['LBFGS_recurse_chunks']:

        print('\n')
        print('L-BFGS, solve across full rollout time recursively, initialize from forward solver in reverse')
        print('\n')

        x_inits = [None for z in range(n_chunks_recursive)]
        times_eb = dt * np.linspace(0, 
                                    T_rollout//n_chunks_recursive, 
                                    res['back_solve_dt_fac'] * (T_rollout//n_chunks_recursive)+1)

        #def exp_bs(x_init): # numpy decorator
        #    return explicit_backsolve(x_init, times_eb=times_eb, fun_eb=fun_eb)

        def exp_bs(x_init): # torch decorator
            x = as_tensor(x_init)
            for t in range(res['back_solve_dt_fac'] *(T_rollout//n_chunks_recursive)):
                x = model_forwarder_eb.forward(x)
            return x.detach().cpu().numpy()

        x_init = get_init(sortL96intoChannels(res['targets_obs'],J=J)[0], res['loss_mask'][0], method='interpolate')
        x_inits[0] = exp_bs(x_init)

        opt_res = optim_initial_state(
            gen,
            T_rollouts=T_rollouts_chunks,
            T_obs=list(np.repeat(T_obs, recursions_per_chunks)),
            N=N,
            n_chunks=n_chunks_recursive,
            optimizer_pars=optimizer_pars,
            x_inits=x_inits,
            targets=[1.*res['targets_obs'] for i in range(n_chunks_recursive)],
            grndtrths=grndtrths_chunks,
            loss_masks=[torch.stack(gen.masks,dim=0) for i in range(n_chunks_recursive)],
            f_init=exp_bs)

        res['x_sols_LBFGS_recurse_chunks'] = opt_res[0]
        res['loss_vals_LBFGS_recurse_chunks'] = opt_res[1]
        res['time_vals_LBFGS_recurse_chunks'] = opt_res[2]
        res['state_mses_LBFGS_recurse_chunks'] = opt_res[3]

        print('\n')
        print('storing results')
        print('\n')
        np.save(res_dir + fn, arr=res)

    # ## L-BFGS, solve across single chunks recursively, initialize from last chunk

    if optimiziation_schemes['LBFGS_chunks']:

        print('\n')
        print('L-BFGS, solve across single chunks recursively, initialize from last chunk')
        print('\n')

        x_inits = [None for i in range(n_chunks_recursive)]
        x_inits[0] = get_init(sortL96intoChannels(res['targets_obs'],J=J)[0], res['loss_mask'][0], method='interpolate')

        opt_res = optim_initial_state(
            gen,
            T_rollouts=np.ones(n_chunks_recursive, dtype=np.int) * (T_rollout//n_chunks_recursive),
            T_obs=list(np.repeat(T_obs, recursions_per_chunks)),
            N=N,
            n_chunks=n_chunks_recursive,
            optimizer_pars=optimizer_pars,
            x_inits=x_inits,
            targets=[1.*res['targets_obs']] + [None for i in range(n_chunks_recursive-1)],
            grndtrths=grndtrths_chunks,
            loss_masks=[torch.stack(gen.masks,dim=0)] + [torch.ones((N,J+1,K)) for i in range(n_chunks_recursive-1)])

        res['x_sols_LBFGS_chunks'] = opt_res[0]
        res['loss_vals_LBFGS_chunks'] = opt_res[1]
        res['time_vals_LBFGS_chunks'] = opt_res[2]
        res['state_mses_LBFGS_chunks'] = opt_res[3]

        print('\n')
        print('storing results')
        print('\n')
        np.save(res_dir + fn, arr=res)


    # ## L-BFGS, solve across full rollout time in one go, initialize from chunked approach

    if optimiziation_schemes['LBFGS_full_chunks']:

        print('\n')
        print('L-BFGS, solve across full rollout time in one go, initialize from chunked approach')
        print('\n')

        x_inits = res['x_sols_LBFGS_chunks'][recursions_per_chunks-1:][::recursions_per_chunks]
        x_inits = [sortL96intoChannels(z,J=J).copy() for z in x_inits]

        opt_res = optim_initial_state(
            gen,
            T_rollouts=T_rollouts,
            T_obs=T_obs,
            N=N,
            n_chunks=n_chunks,
            optimizer_pars=optimizer_pars,
            x_inits=x_inits,
            targets=targets,
            grndtrths=grndtrths,
            loss_masks=loss_masks)

        res['x_sols_LBFGS_full_chunks'] = opt_res[0]
        res['loss_vals_LBFGS_full_chunks'] = opt_res[1]
        res['time_vals_LBFGS_full_chunks'] = opt_res[2]
        res['state_mses_LBFGS_full_chunks'] = opt_res[3]

        print('\n')
        print('storing results')
        print('\n')
        np.save(res_dir + fn, arr=res)


    # ## numerical forward solve in reverse

    if optimiziation_schemes['backsolve']:

        print('\n')
        print('numerical forward solve in reverse')
        print('\n')

        res['state_mses_backsolve'] = np.zeros((n_chunks_recursive, len(n_starts)))
        res['time_vals_backsolve'] = np.zeros((n_chunks_recursive, len(n_starts)))
        res['x_sols_backsolve'] = np.zeros((n_chunks_recursive, len(n_starts), K*(J+1)))
        res['loss_vals_backsolve'] = np.zeros((n_chunks_recursive, len(n_starts)))

        times_eb = dt * np.linspace(0, T_rollouts[0], res['back_solve_dt_fac'] * T_rollouts[0]+1)
        print('backward solving')
        res['time_vals_backsolve'][0] = time.time()
        x_init = get_init(sortL96intoChannels(res['targets_obs'][0],J=J), res['loss_mask'][0], method='interpolate')
        res['x_sols_backsolve'][0] = explicit_backsolve(sortL96fromChannels(x_init), times_eb, fun_eb)
        res['time_vals_backsolve'][0] = time.time() - res['time_vals_backsolve'][0]

        for j in range(n_chunks_recursive):
            res['x_sols_backsolve'][j] = res['x_sols_backsolve'][0]
            res['time_vals_backsolve'][j] = res['time_vals_backsolve'][0]
            res['state_mses_backsolve'][j] = ((res['x_sols_backsolve'][j] - grndtrths_chunks[j])**2).mean(axis=1)

        print('\n')
        print('storing results')
        print('\n')
        np.save(res_dir + fn, arr=res)


    # ## L-BFGS, solve across full rollout time in one go, initiate from backward solution

    if optimiziation_schemes['LBFGS_full_backsolve']:

        print('\n')
        print('L-BFGS, solve across full rollout time in one go, initiate from backward solution')
        print('\n')

        x_inits = res['x_sols_backsolve'][recursions_per_chunks-1:][::recursions_per_chunks]
        x_inits = [sortL96intoChannels(z,J=J).copy() for z in x_inits]

        opt_res = optim_initial_state(
            gen,
            T_rollouts=T_rollouts,
            T_obs=T_obs,
            N=N,
            n_chunks=n_chunks,
            optimizer_pars=optimizer_pars,
            x_inits=x_inits,
            targets=targets,
            grndtrths=grndtrths,
            loss_masks=loss_masks)

        res['x_sols_LBFGS_full_backsolve'] = opt_res[0]
        res['loss_vals_LBFGS_full_backsolve'] = opt_res[1]
        res['time_vals_LBFGS_full_backsolve'] = opt_res[2]
        res['state_mses_LBFGS_full_backsolve'] = opt_res[3]

        print('\n')
        print('storing results')
        print('\n')
        np.save(res_dir + fn, arr=res)


    # ## L-BFGS, solve across full rollout time in one go
    # - warning, this can be excruciatingly slow and hard to converge !

    if optimiziation_schemes['LBFGS_full_persistence']:

        print('\n')
        print('L-BFGS, solve across full rollout time in one go')
        print('\n')

        x_init = get_init(sortL96intoChannels(res['targets_obs'],J=J)[0], res['loss_mask'][0], method='interpolate')
        x_inits = [x_init for j in range(n_chunks)]

        opt_res = optim_initial_state(
            gen,
            T_rollouts=T_rollouts,
            T_obs=T_obs,
            N=N,
            n_chunks=n_chunks,
            optimizer_pars=optimizer_pars,
            x_inits=x_inits,
            targets=targets,
            grndtrths=grndtrths,
            loss_masks=loss_masks)

        res['x_sols_LBFGS_full_persistence'] = opt_res[0]
        res['loss_vals_LBFGS_full_persistence'] = opt_res[1]
        res['time_vals_LBFGS_full_persistence'] = opt_res[2]
        res['state_mses_LBFGS_full_persistence'] = opt_res[3]

        print('\n')
        print('storing results')
        print('\n')
        np.save(res_dir + fn, arr=res)
    
    print('\n')
    print('done')
    print('\n')


def get_model(model_pars, res_dir, exp_dir=''):

    if model_pars['exp_id'] is None: 
        # loading 'perfect' (up to machine-precision-level quirks) L96 model in pytorch
        args = {'filters': [0],
               'kernel_sizes': [4],
               'init_net': 'analytical',
               'K_net': model_pars['K_net'],
               'J_net': model_pars['J_net'],
               'dt_net': model_pars['dt_net'],
               'model_forwarder': model_pars['model_forwarder']}

        model, model_forwarder = named_network(
            model_name='MinimalConvNetL96',
            n_input_channels=model_pars['J_net']+1,
            n_output_channels=model_pars['J_net']+1,
            seq_length=1,
            **args
        )
        if model_pars['J_net'] > 0:
            args['conf_exp'] = '00_analyticalMinimalConvNet'
        else:
            args['conf_exp'] = '00_analyticalMinimalConvNet_oneLevel'
    else:

        exp_names = os.listdir(exp_dir + 'experiments/')
        conf_exp = exp_names[np.where(np.array([name[:2] for name in exp_names])==str(model_pars['exp_id']))[0][0]][:-4]
        print('conf_exp', conf_exp)

        # ### pick a (trained) emulator

        args = setup(conf_exp=f'experiments/{conf_exp}.yml')
        
        # ### choose numerical solver scheme

        args['model_forwarder'] = model_pars['model_forwarder']
        args['dt_net'] = model_pars['dt_net']

        # ### load & instantiate the emulator

        model, model_forwarder, _ = load_model_from_exp_conf(res_dir, args)

    # switch off parameter gradients for model:
    for p in model.parameters():
        p.requires_grad = False

    return model, model_forwarder, args


def get_init(x_init, obs_mask=None, method='interpolate'):

    N, J, K = x_init.shape
    J -= 1

    obs_mask = 1.*(x_init==0.) if obs_mask is None else obs_mask

    assert x_init.shape == obs_mask.shape
    assert J == 0

    x_p = np.zeros_like(x_init)

    if method == 'interpolate':

        K_range = np.arange(K)
        for n in range(N):
            for j in range(J+1):
                mask_nj = np.where(obs_mask[n][j]>0.)[0]
                x_p[n,j] = np.interp(K_range, mask_nj, x_init[n, j, mask_nj], period=K)

    else: 
        raise NotImplementedError()

    return x_p
