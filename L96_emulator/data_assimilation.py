import numpy as np
import torch 

import os
import time

from L96_emulator.util import dtype, dtype_np, device
from L96_emulator.util import sortL96fromChannels, sortL96intoChannels
from L96_emulator.util import predictor_corrector, rk4_default, get_data

from L96_emulator.run import setup, sel_dataset_class

from L96_emulator.eval import load_model_from_exp_conf, get_rollout_fun, named_network, Rollout

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
        if m is None:
            return self.ndistr.log_prob(x - y).sum() # sum from iid over dims
        assert y.shape == m.shape and x.shape == y.shape
        return (m * self.ndistr.log_prob(x - y)).sum() # sum from iid over dims


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
        if m is None:
            m = self.mask
        assert y.shape == m.shape and x.shape == y.shape
        return (m * self.ndistr.log_prob(x - y)).sum() # sum from iid over dims


class GenModel(torch.nn.Module):

    def __init__(self, model_forwarder, model_observer, prior, 
                 T=1, T_obs=None, x_init=None):

        super(GenModel, self).__init__()

        self.model_forwarder = model_forwarder
        self.T = T

        self.model_observer = model_observer
        self.T_obs = [T-1] if T_obs is None else T_obs

        self.prior = prior        
        x_init = self.prior.sample() if x_init is None else x_init        

        assert x_init.ndim in [2,3]
        # variable container for e.g. maximim-likelihood estimate: 
        self.X = torch.nn.Parameter(torch.as_tensor(x_init, device=device, dtype=dtype))

    def forward(self, x=None, T=None):

        x = self.X if x is None else x
        T = self.T if T is None else T
        y = []
        for t in range(self.T):
            x = self.model_forwarder.forward(x)
            if t in self.T_obs:
                y.append(self.model_observer.forward(x))
        return torch.stack(y)

    def sample(self, x=None, m=None, T=None):

        x = self.X if x is None else x
        T = self.T if T is None else T
        y = []
        for t in range(self.T):
            x = self.model_forwarder.forward(x)
            if t in self.T_obs:
                y.append(self.model_observer.sample(x, m=m))
        return torch.stack(y) if len(self.T_obs) > 1 else y[0]

    def log_prob(self, y, x=None, m=None, T=None):

        log_probs = torch.stack([self.model_observer.log_prob(y_fwd, y, m=m) for y_fwd in self.forward(x, T)])
        return log_probs.sum() # assuming observations are independent


def mse_loss_fullyObs(x, t):

    assert x.shape == t.shape
    return ((x - t)**2).mean()


def mse_loss_masked(x, t, m):

    assert x.shape == m.shape and t.shape == m.shape
    return (m * ((x - t)**2)).sum() / m.sum()


def optim_initial_state(
      model_forwarder, model_observer, prior, 
      T_rollouts, T_obs, N, n_chunks,
      n_steps, optimizer_pars,
      x_inits, targets, grndtrths, 
      loss_masks=None, f_init=None):

    sample_shape = prior.sample().shape # (..., J+1, K)
    J, K = sample_shape[-2]-1, sample_shape[-1]
    
    x_sols = np.zeros((n_chunks, N, K*(J+1)))
    loss_vals = np.zeros((n_steps,N))
    time_vals = time.time() * np.ones((n_steps,N))
    state_mses = np.zeros(n_chunks)
    
    loss_masks = [torch.ones((N,J+1,K)) for i in range(n_chunks)] if loss_masks is None else loss_masks
    assert len(loss_masks) == n_chunks

    i_ = 0
    for j in range(n_chunks):

        print('\n')
        print(f'optimizing over chunk #{j+1} out of {n_chunks}')
        print('\n')

        target = sortL96intoChannels(as_tensor(targets[j]),J=J)

        for n in range(N):

            print('\n')
            print(f'optimizing over initial state #{n+1} / {N}')
            print('\n')

            gen = GenModel(model_forwarder, model_observer, prior, T=T_rollouts[j], 
                           T_obs=T_obs[j], x_init = x_inits[j][n:n+1])

            optimizer = torch.optim.LBFGS(params=[gen.X],
                                          lr=optimizer_pars['lr'],
                                          max_iter=optimizer_pars['max_iter'],
                                          max_eval=optimizer_pars['max_eval'],
                                          tolerance_grad=optimizer_pars['tolerance_grad'],
                                          tolerance_change=optimizer_pars['tolerance_change'],
                                          history_size=optimizer_pars['history_size'],
                                          line_search_fn='strong_wolfe')

            i_n = 0
            for i in range(n_steps//n_chunks):

                with torch.no_grad():
                    loss = - gen.log_prob(y=target[n:n+1], m=loss_masks[j][n:n+1])
                    if torch.isnan(loss):
                        loss_vals[i_n,n] = loss.detach().cpu().numpy()
                        i_n += 1
                        continue

                def closure():
                    loss = - gen.log_prob(y=target[n:n+1], m=loss_masks[j][n:n+1])
                    optimizer.zero_grad()
                    loss.backward()
                    return loss
                optimizer.step(closure)
                loss_vals[i_n,n] = loss.detach().cpu().numpy()
                time_vals[i_n,n] = time.time() - time_vals[i_n,n]
                print((time_vals[i_n,n], loss_vals[i_n,n]))
                i_n += 1

            x_sols[j][n] = sortL96fromChannels(gen.X.detach().cpu().numpy().copy())

        i_ += i_n

        if j < n_chunks - 1 and targets[j+1] is None:
            targets[j+1] = x_sols[j].copy()
        state_mses[j] = ((x_sols[j] - grndtrths[j])**2).mean()

        with torch.no_grad():  
            print('Eucl. distance to initial value', mse_loss_fullyObs(x_sols[j], grndtrths[j]))
            print('Eucl. distance to x_init', mse_loss_fullyObs(x_sols[j], sortL96fromChannels(x_inits[j])))
            if loss_masks[j] is None:
                print('Eucl. distance to target', mse_loss_fullyObs(x_sols[j], targets[j]))
            else:
                print('Eucl. distance to target', mse_loss_masked(x_sols[j], 
                                                            targets[j],
                                                            sortL96fromChannels(loss_masks[j]).detach().cpu().numpy()))

        if j < n_chunks - 1 and x_inits[j+1] is None:
            x_inits[j+1] = sortL96intoChannels(x_sols[j], J=J).copy()
            if not f_init is None:
                x_inits[j+1] = f_init(x_inits[j+1])

    return x_sols, loss_vals, time_vals, state_mses


def solve_initstate(system_pars, model_pars, optimizer_pars, setup_pars, res_dir, data_dir):

    K,J,T,dt,N_trials=system_pars['K'],system_pars['J'],system_pars['T'],system_pars['dt'],system_pars['N_trials']
    spin_up_time,train_frac,normalize_data=system_pars['spin_up_time'],system_pars['train_frac'],system_pars['normalize_data']
    F,h,b,c=system_pars['F'],system_pars['h'],system_pars['b'],system_pars['c']
    obs_operator, obs_operator_args=system_pars['obs_operator'], system_pars['obs_operator_args'] 
    
    n_starts, T_rollout, n_chunks, N = setup_pars['n_starts'], setup_pars['T_rollout'], setup_pars['n_chunks'], setup_pars['N']
    n_steps, prediction_task, lead_time = setup_pars['n_steps'], setup_pars['prediction_task'], setup_pars['lead_time']
    
    exp_id, model_forwarder, dt_net = model_pars['exp_id'], model_pars['model_forwarder'], model_pars['dt_net']
    
    back_solve_dt_fac = optimizer_pars['back_solve_dt_fac']
    
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

    # function for explicitly solving backwards
    dX_dt = np.empty(K*(J+1), dtype=dtype_np)
    if J > 0:
        def fun_eb(t, x):
            return - f2(x, F, h, b, c, dX_dt, K, J)
    else:
        def fun_eb(t, x):
            return - f1(x, F, dX_dt, K)

    def model_eb(t, x):
        return - sortL96fromChannels(model.forward(sortL96intoChannels(x,J=J)))


    # # Solving a fully-observed inverse problem

    # ### load / simulate data

    out, datagen_dict = get_data(K=K, J=J, T=T, dt=dt, N_trials=N_trials, F=F, h=h, b=b, c=c, 
                                 resimulate=True, solver=rk4_default,
                                 save_sim=False, data_dir=data_dir)

    DatasetClass = sel_dataset_class(prediction_task=prediction_task,N_trials=1)
    dg_train = DatasetClass(data=out, J=J, offset=lead_time, normalize=normalize_data, 
                       start=int(spin_up_time/dt), 
                       end=int(np.floor(out.shape[0]*train_frac)))

    # ### instantiate observation operator

    model_observer = obs_operator(**obs_operator_args)
    
    target_obs = model_observer.sample(sortL96intoChannels(as_tensor(out[n_starts+T_rollout]),J=J))
    target_obs = sortL96fromChannels(target_obs.detach().cpu().numpy())

    # ### define prior over initial states

    prior = torch.distributions.normal.Normal(loc=torch.zeros((1,J+1,K)), 
                                              scale=1.*torch.ones((1,J+1,K)))


    # ## L-BFGS, solve across full rollout time in one go
    # - warning, this can be excruciatingly slow and hard to converge !

    print('\n')
    print('L-BFGS, solve across full rollout time in one go')
    print('\n')

    T_rollouts = np.arange(1, n_chunks+1) * (T_rollout//n_chunks)
    T_obs = [[t - 1] for t in T_rollouts]
    grndtrths = [out[n_starts+T_rollout-(j+1)*(T_rollout//n_chunks)] for j in range(n_chunks)]
    targets = [1.*target_obs for i in range(n_chunks)]
    loss_masks = [1.*model_observer.mask for i in range(n_chunks)]

    x_init = sortL96intoChannels(out[n_starts+T_rollout], J=J)
    x_inits = [x_init.copy() for j in range(n_chunks)]

    res = optim_initial_state(
      model_forwarder, model_observer, prior, 
      T_rollouts, T_obs, N, n_chunks,
      n_steps, optimizer_pars,
      x_inits, targets, grndtrths, 
      loss_masks=loss_masks)

    x_sols_LBFGS_full_persistence, loss_vals_LBFGS_full_persistence = res[0], res[1]
    time_vals_LBFGS_full_persistence, state_mses_LBFGS_full_persistence = res[2], res[3]


    # ## L-BFGS, solve across full rollout time recursively, initialize from  explicit backward solution

    print('\n')
    print('L-BFGS, solve across full rollout time recursively, initialize from explicit backward solution')
    print('\n')

    T_rollouts = np.arange(1, n_chunks+1) * (T_rollout//n_chunks)
    T_obs = [[t - 1] for t in T_rollouts]
    grndtrths = [out[n_starts+T_rollout-(j+1)*(T_rollout//n_chunks)] for j in range(n_chunks)]
    x_inits = [None for z in range(n_chunks)]

    times_eb = dt * np.linspace(0, T_rollout//n_chunks, back_solve_dt_fac * (T_rollout//n_chunks)+1)
    def explicit_backsolve(x_init):
        x_init = sortL96fromChannels(x_init)
        x_sols = np.zeros_like(x_init)
        for i__ in range(x_init.shape[0]):
            out2 = rk4_default(fun=fun_eb, y0=x_init[i__], times=times_eb)
            x_sols[i__] = out2[-1].copy()#.detach().cpu().numpy().copy()
        return sortL96intoChannels(x_sols,J=J)
    x_inits[0] = explicit_backsolve(sortL96intoChannels(target_obs,J=J))

    targets = [1.*target_obs for i in range(n_chunks)]
    loss_masks = [1.*model_observer.mask for i in range(n_chunks)]
    
    res = optim_initial_state(
      model_forwarder, model_observer, prior, 
      T_rollouts, T_obs, N, n_chunks,
      n_steps, optimizer_pars,
      x_inits, targets, grndtrths, 
      loss_masks=loss_masks, f_init=explicit_backsolve)
    
    x_sols_LBFGS_recurse_chunks, loss_vals_LBFGS_recurse_chunks = res[0], res[1]
    time_vals_LBFGS_recurse_chunks, state_mses_LBFGS_recurse_chunks = res[2], res[3]

    # ## L-BFGS, solve across single chunks recursively, initialize from last chunk

    print('\n')
    print('L-BFGS, solve across single chunks recursively, initialize from last chunk')
    print('\n')

    T_rollouts = np.ones(n_chunks, dtype=np.int) * (T_rollout//n_chunks)
    x_inits, targets = [None for i in range(n_chunks)], [None for i in range(n_chunks)]
    x_inits[0] = sortL96intoChannels(np.atleast_2d(out[n_starts+T_rollout].copy()),J=J)
    targets[0] = 1.*target_obs
    loss_masks = [None for i in range(n_chunks)]
    loss_masks[0] = 1.*obs.mask

    res = optim_initial_state(
      model_forwarder, model_observer, prior, 
      T_rollouts, T_obs, N, n_chunks,
      n_steps, optimizer_pars,
      x_inits, targets, grndtrths, 
      loss_masks=loss_masks)

    x_sols_LBFGS_chunks, loss_vals_LBFGS_chunks = res[0], res[1]
    time_vals_LBFGS_chunks, state_mses_LBFGS_chunks = res[2], res[3]


    # ## L-BFGS, solve across full rollout time in one go, initialize from chunked approach

    print('\n')
    print('L-BFGS, solve across full rollout time in one go, initialize from chunked approach')
    print('\n')

    T_rollouts = np.arange(1, n_chunks+1) * (T_rollout//n_chunks)
    x_inits = [sortL96intoChannels(z,J=J).copy() for z in x_sols_LBFGS_chunks] 
    targets, loss_masks = [1.*target_obs for i in range(n_chunks)], [1.*obs.mask for i in range(n_chunks)]

    res = optim_initial_state(
      model_forwarder, model_observer, prior, 
      T_rollouts, T_obs, N, n_chunks,
      n_steps, optimizer_pars,
      x_inits, targets, grndtrths, 
      loss_masks=loss_masks)

    x_sols_LBFGS_full_chunks, loss_vals_LBFGS_full_chunks  = res[0], res[1]
    time_vals_LBFGS_full_chunks, state_mses_LBFGS_full_chunks = res[2], res[3]


    # ## L-BFGS, solve across full rollout time in one go, initiate from backward solution

    print('\n')
    print('L-BFGS, solve across full rollout time in one go, initiate from backward solution')
    print('\n')

    state_mses_backsolve = np.zeros(n_chunks)
    time_vals_backsolve = np.zeros(n_chunks)

    x_sols_backsolve = np.zeros((n_chunks, len(n_starts), K*(J+1)))
    i_ = 0
    for j in range(n_chunks):

        T_i = T_rollouts[j]
        times = dt * np.linspace(0, T_i, back_solve_dt_fac * T_i+1) # note the increase in temporal resolution!
        print('backward solving')
        time_vals_backsolve[j] = time.time()
        x_init = np.zeros((len(n_starts), K*(J+1)))
        for i__ in range(len(n_starts)):
            out2 = rk4_default(fun=fun_eb, y0=out[n_starts[i__]+T_rollout].copy(), times=times)
            x_init[i__] = out2[-1].copy()
        x_sols_backsolve[j] = x_init.copy()
        time_vals_backsolve[j] = time.time() - time_vals_backsolve[j]
        state_mses_backsolve[j] = ((x_sols_backsolve[j] - out[n_starts+T_rollout-(j+1)*T_rollout//n_chunks])**2).mean()

    x_inits = [sortL96intoChannels(z,J=J).copy() for z in x_sols_backsolve]
    res = optim_initial_state(
      model_forwarder, model_observer, prior,
      T_rollouts, T_obs, N, n_chunks,
      n_steps, optimizer_pars,
      x_inits, targets, grndtrths,
      loss_masks=loss_masks)

    x_sols_LBFGS_full_backsolve, loss_vals_LBFGS_full_backsolve = res[0], res[1]
    time_vals_LBFGS_full_backsolve, state_mses_LBFGS_full_backsolve = res[2], res[3]


    # ## plot and compare results

    print('\n')
    print('done, storing results')
    print('\n')

    initial_states = [out[n_starts+j*T_rollout//n_chunks] for j in range(n_chunks)]
    initial_states = np.stack([sortL96intoChannels(z,J=J) for z in initial_states])

    model_forwarder_str = args['model_forwarder']
    optimizer_str = optimizer_pars['optimizer']
    obs_operator_str = obs.__class__.__name__
    fn = 'results/data_assimilation/fullyobs_initstate_tests_'
    fn = fn + f'exp{exp_id}_{model_forwarder_str}_{optimizer_str}_{obs_operator_str}'
    np.save(res_dir + fn,
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

                'optimizer_pars' : optimizer_pars, 

                'targets' : sortL96intoChannels(out[n_starts+T_rollout], J=J),
                'initial_states' : initial_states,
                'loss_masks' : loss_masks,
                
                'obs_operator' : obs_operator_str,
                'obs_operator_args' : obs_operator_args,

                'loss_vals_LBFGS_full_backsolve' : loss_vals_LBFGS_full_backsolve, 
                'loss_vals_LBFGS_full_persistence' : loss_vals_LBFGS_full_persistence,
                'loss_vals_LBFGS_full_chunks' : loss_vals_LBFGS_full_chunks,
                'loss_vals_LBFGS_chunks' : loss_vals_LBFGS_chunks,
                'loss_vals_LBFGS_recurse_chunks' : loss_vals_LBFGS_recurse_chunks,

                'time_vals_LBFGS_full_backsolve' :   time_vals_LBFGS_full_backsolve,
                'time_vals_LBFGS_full_persistence' : time_vals_LBFGS_full_persistence,
                'time_vals_LBFGS_full_chunks' :      time_vals_LBFGS_full_chunks,
                'time_vals_LBFGS_chunks' :           time_vals_LBFGS_chunks,
                'time_vals_backsolve' :              time_vals_backsolve,
                'time_vals_LBFGS_recurse_chunks' :   time_vals_LBFGS_recurse_chunks, 

                'state_mses_LBFGS_full_backsolve' :   state_mses_LBFGS_full_backsolve,
                'state_mses_LBFGS_full_persistence' : state_mses_LBFGS_full_persistence,
                'state_mses_LBFGS_full_chunks' :      state_mses_LBFGS_full_chunks,
                'state_mses_LBFGS_chunks' :           state_mses_LBFGS_chunks,
                'state_mses_backsolve' :              state_mses_backsolve,
                'state_mses_LBFGS_recurse_chunks' :   state_mses_LBFGS_recurse_chunks,

                'x_sols_LBFGS_full_backsolve' : x_sols_LBFGS_full_backsolve, 
                'x_sols_LBFGS_full_persistence' : x_sols_LBFGS_full_persistence,
                'x_sols_LBFGS_full_chunks' : x_sols_LBFGS_full_chunks,
                'x_sols_LBFGS_chunks' : x_sols_LBFGS_chunks,
                'x_sols_LBFGS_recurse_chunks' : x_sols_LBFGS_recurse_chunks, 
                'x_sols_backsolve' : x_sols_backsolve
                })
