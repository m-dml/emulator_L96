import numpy as np
import torch 

import os
import time

from L96_emulator.util import dtype, dtype_np, device, as_tensor
from L96_emulator.util import sortL96fromChannels, sortL96intoChannels
from L96_emulator.util import predictor_corrector, rk4_default, get_data

from L96_emulator.run import setup, sel_dataset_class

from L96_emulator.eval import load_model_from_exp_conf, get_rollout_fun, named_network, Rollout

from L96_emulator.networks import Model_forwarder_predictorCorrector, Model_forwarder_rk4default

from L96_emulator.likelihood import ObsOp_identity, ObsOp_subsampleGaussian, GenModel, SimplePrior


def mse_loss_fullyObs(x, t):

    assert x.shape == t.shape
    return ((x - t)**2).mean()


def mse_loss_masked(x, t, m):

    assert x.shape == m.shape and t.shape == m.shape
    return (m * ((x - t)**2)).sum() / m.sum()


def convergenced_LBFGS(state_dict):

    return False


def optim_initial_state(
    gen,
    T_rollouts,
    T_obs,
    N,
    n_chunks,
    optimizer_pars,
    x_inits,
    targets,
    grndtrths=None,
    loss_masks=None,
    priors=None,
    f_init=None):

    sample_shape = gen.prior.sample().shape # (..., J+1, K)
    J, K = sample_shape[-2]-1, sample_shape[-1]
    n_steps = optimizer_pars['n_steps']

    x_sols = np.zeros((n_chunks, N, K*(J+1)))
    loss_vals = np.inf * np.ones((n_steps*n_chunks,N))
    time_vals = time.time() * np.ones((n_steps*n_chunks,N))
    state_mses = np.inf * np.ones((n_chunks, N))
    
    loss_masks = [torch.ones((N,J+1,K)) for i in range(n_chunks)] if loss_masks is None else loss_masks
    assert len(loss_masks) == n_chunks
    
    if priors is None:
        class Const_prior(object):
            def __init__(self):
                pass
            def log_prob(self, x):
                return 0.
        priors = [Const_prior() for n in range(N)] 
    assert len(priors) == N
    
    i_ = 0
    for j in range(n_chunks):

        print('\n')
        print(f'optimizing over chunk #{j+1} out of {n_chunks}')
        print('\n')

        target = sortL96intoChannels(as_tensor(targets[j]),J=J)
        loss_mask = loss_masks[j]
        
        assert len(target) == len(T_obs[j]) and len(loss_mask) == len(T_obs[j])

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
                    loss = - gen.log_prob(y=target[:,n:n+1],
                                          m=loss_mask[:,n:n+1],
                                          T_obs=T_obs[j])
                    loss = loss - priors[n].log_prob(gen.X)

                    if i_n == 0:
                        print('initial loss: ', loss)
                    if torch.any(torch.isnan(loss)):
                        loss_vals[i_n,n] = loss.detach().cpu().numpy()
                        time_vals[i_+i_n,n] = time.time() - time_vals[i_+i_n,n]
                        print(('{:.4f}'.format(time_vals[i_n,n]), loss_vals[i_n,n]))
                        print('NaN loss - skipping iteration')

                        print('optimizier.state', optimizer.state[gen.X])

                        continue

                def closure():
                    loss = - gen.log_prob(y=target[:,n:n+1],
                                          m=loss_mask[:,n:n+1],
                                          T_obs=T_obs[j])
                    if not priors is None:
                        loss = loss - priors[n].log_prob(gen.X)
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    if loss.requires_grad:
                        loss.backward()
                    return loss

                optimizer.step(closure)
                loss_vals[i_+i_n,n] = loss.detach().cpu().numpy()
                time_vals[i_+i_n,n] = time.time() - time_vals[i_+i_n,n]
                print(('{:.4f}'.format(time_vals[i_n,n]), loss_vals[i_n,n]))

            x_sols[j][n] = sortL96fromChannels(gen.X.detach().cpu().numpy().copy())
            state_mses[j][n] = np.inf if grndtrths is None else ((x_sols[j][n] - grndtrths[j][n])**2).mean()

        # if solving recursively, define next target as current initial state estimate 
        if j < n_chunks - 1 and targets[j+1] is None:
            targets[j+1] = x_sols[j].copy().reshape(1, *x_sols[j].shape)

        i_ += n_steps

        with torch.no_grad():  
            if not grndtrths is None:
                print('Eucl. distance to initial value', mse_loss_fullyObs(x_sols[j], grndtrths[j]))
            print('Eucl. distance to x_init', mse_loss_fullyObs(x_sols[j], sortL96fromChannels(x_inits[j])))

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


def solve_initstate(system_pars, model_pars, optimizer_pars, setup_pars, optimiziation_schemes,
                    res_dir, data_dir, fn=None):

    # extract key variable names from input dicts
    K, J = system_pars['K'], system_pars['J']
    T, dt, N_trials = system_pars['T'], system_pars['dt'], system_pars['N_trials']

    n_starts, T_rollout, T_pred = setup_pars['n_starts'], setup_pars['T_rollout'], setup_pars['T_pred'] 
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
    obs_operator_str = model_observer.__class__.__name__
    exp_id = model_pars['exp_id']

    # output dictionary
    res = { 'exp_id' : exp_id,
            'K' : K,
            'J' : J,
            'T' : T, 
            'dt' : dt,

            'back_solve_dt_fac' : system_pars['back_solve_dt_fac'],
            'F' : system_pars['F'], 
            'h' : system_pars['h'], 
            'b' : system_pars['b'], 
            'c' : system_pars['c'],

            'conf_exp' : args['conf_exp'],
            'model_forwarder' : model_pars['model_forwarder'], # should still be string
            'dt_net' : model_pars['dt_net'],

            'n_starts' : n_starts,
            'T_rollout' : T_rollout,
            'T_pred' : T_pred, 
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
    
    # ## Generate observed data: (sub-)sample noisy observations
    res['test_state'] = sortL96intoChannels(out[n_starts+T_pred], J=J).reshape(1, len(n_starts), J+1, K)
    res['test_state_obs'] = gen._sample_obs(as_tensor(res['test_state'])) # sets the loss masks!
    res['test_state_obs'] = sortL96fromChannels(res['test_state_obs'].detach().cpu().numpy())
    res['test_state_mask'] = torch.stack(gen.masks,dim=0).detach().cpu().numpy()

    res['targets_obs'] = gen._sample_obs(as_tensor(res['targets'])) # sets the loss masks!
    res['targets_obs'] = sortL96fromChannels(res['targets_obs'].detach().cpu().numpy())
    res['loss_mask'] = torch.stack(gen.masks,dim=0).detach().cpu().numpy()


    if fn is None:
        fn = 'results/data_assimilation/fullyobs_initstate_tests_'
        fn = fn + f'exp{exp_id}_{model_forwarder_str}_{optimizer_str}_{obs_operator_str}'

    print('\n')
    print('storing intermediate results')
    print('\n')
    np.save(res_dir + fn, arr=res)

    # ### define setup for optimization

    T_rollouts = np.arange(1, n_chunks+1) * (T_rollout//n_chunks)
    grndtrths = [out[n_starts] for j in range(n_chunks)]
    targets = [res['targets_obs'][:len(T_obs[j])] for j in range(n_chunks)]
    loss_masks = [torch.stack(gen.masks[:len(T_obs[j])],dim=0) for j in range(n_chunks)]
    
    grndtrths_chunks = [out[n_starts] for j in range(n_chunks_recursive)]
    
    # ## L-BFGS, solve across full rollout time recursively, initialize from forward solver in reverse

    if optimiziation_schemes['LBFGS_recurse_chunks']:

        print('\n')
        print('L-BFGS, solve across full rollout time recursively, initialize from forward solver in reverse')
        print('\n')

        assert len(T_obs) == 1 # only allow single observation at end of interval
        assert len(res['targets_obs']) == 1

        # functions for explicitly solving backwards
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

        def exp_fsr(x_init): # torch decorator for forward-solve in reverse learned from model_exp_id
            x = as_tensor(x_init)
            for t in range(res['back_solve_dt_fac'] * T_rollout //n_chunks_recursive):
                x = model_forwarder_eb.forward(x)
            return x.detach().cpu().numpy()

        x_init = get_init(sortL96intoChannels(res['targets_obs'],J=J)[0], 
                          res['loss_mask'][0], 
                          method='interpolate')
        x_inits = [exp_fsr(x_init)] +  [None for j in range(n_chunks_recursive)]

        opt_res = optim_initial_state(
            gen,
            T_rollouts=np.arange(1, n_chunks_recursive+1) * (T_rollout//n_chunks_recursive),
            T_obs=[[j] for j in range(recursions_per_chunks)],
            N=N,
            n_chunks=n_chunks_recursive,
            optimizer_pars=optimizer_pars,
            x_inits=x_inits,
            targets=list(np.repeat(targets, recursions_per_chunks, axis=0)),
            grndtrths=[out[n_starts+T_rollout-(j+1)*(T_rollout//n_chunks_recursive)] for j in range(n_chunks_recursive)],
            loss_masks=list(np.repeat(loss_masks, recursions_per_chunks, axis=0)),
            f_init=exp_fsr)

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

        assert len(T_obs) == 1 # only allow single observation at end of interval
        assert len(res['targets_obs']) == 1

        x_inits = [None for i in range(n_chunks_recursive)]
        x_inits[0] = get_init(sortL96intoChannels(res['targets_obs'],J=J)[0], res['loss_mask'][0], method='interpolate')
        
        opt_res = optim_initial_state(
            gen,
            T_rollouts=np.ones(n_chunks_recursive, dtype=np.int) * (T_rollout//n_chunks_recursive),
            T_obs=[[0] for j in range(n_chunks_recursive)],
            N=N,
            n_chunks=n_chunks_recursive,
            optimizer_pars=optimizer_pars,
            x_inits=x_inits,
            targets=[res['targets_obs']] + [None for i in range(n_chunks_recursive-1)],
            grndtrths=[out[n_starts+T_rollout-(j+1)*(T_rollout//n_chunks_recursive)] for j in range(n_chunks_recursive)],
            loss_masks=[torch.stack(gen.masks,dim=0)] + [torch.ones((1,N,J+1,K)) for i in range(n_chunks_recursive-1)])

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

        # functions for explicitly solving backwards
        from L96sim.L96_base import f1, f2, pf2

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

        res['state_mses_backsolve'] = np.zeros((n_chunks_recursive, len(n_starts)))
        res['x_sols_backsolve'] = np.zeros((n_chunks_recursive, len(n_starts), K*(J+1)))
        res['loss_vals_backsolve'] = np.zeros((n_chunks_recursive, len(n_starts)))

        print('backward solving')
        res['time_vals_backsolve'] = time.time() *  np.ones((n_chunks_recursive, len(n_starts)))

        x_init = get_init(sortL96intoChannels(res['targets_obs'][0],J=J), res['loss_mask'][0], method='interpolate')
        for j in range(recursions_per_chunks):
            times_eb = dt * np.linspace(0, 
                                        T_rollouts[0] / recursions_per_chunks, 
                                        res['back_solve_dt_fac'] * T_rollouts[0] / recursions_per_chunks + 1)
            print('x_init.shape', sortL96fromChannels(x_init).shape)
            res['x_sols_backsolve'][j] = explicit_backsolve(sortL96fromChannels(x_init), times_eb, fun_eb)
            x_init = sortL96intoChannels(res['x_sols_backsolve'][j].copy(), J=J)
            print('x_init.shape - out', x_init.shape)
            res['time_vals_backsolve'][j] = res['time_vals_backsolve'][0]
            x_target = out[n_starts+T_rollouts[0]-(j+1)*(T_rollouts[0]//recursions_per_chunks)]
            res['state_mses_backsolve'][j] = ((res['x_sols_backsolve'][j] - x_target)**2).mean(axis=1)
            res['time_vals_backsolve'][j] = time.time() - res['time_vals_backsolve'][0]
        for j in range(recursions_per_chunks, n_chunks_recursive):
            res['x_sols_backsolve'][j] = res['x_sols_backsolve'][recursions_per_chunks-1]
            res['time_vals_backsolve'][j] = res['time_vals_backsolve'][recursions_per_chunks-1]
            res['state_mses_backsolve'][j] = ((res['x_sols_backsolve'][j] - out[n_starts])**2).mean(axis=1)

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


def solve_4dvar(y, m, T_obs, T_win, x_init, model_pars, obs_pars, optimizer_pars, res_dir):
    """
    def solve_4dvar(system_pars, model_pars, optimizer_pars, setup_pars, optimiziation_schemes,
                res_dir, data_dir, fn=None):
    """
    
    # extract key variable names from input dicts
    T, N, J, K = m.shape
    J -= 1
    assert y.shape == (T,N,(J+1)*K)

    if x_init is None:
        x_init = get_init(sortL96intoChannels(y[0],J=J).detach().cpu(), m[0].detach().cpu(), method='interpolate')
    assert x_init.shape == (N, J+1, K)
    
    # get model
    model, model_forwarder, args = get_model(model_pars, res_dir=res_dir, exp_dir='')
    model_observer = obs_pars['obs_operator'](**obs_pars['obs_operator_args'])
    prior = torch.distributions.normal.Normal(loc=torch.zeros((1,J+1,K)), 
                                              scale=1.*torch.ones((1,J+1,K)))
    gen = GenModel(model_forwarder, model_observer, prior, T=T_win, x_init=None)
    priors = None

    assert len(T_obs) == T
    n_starts = np.max(T_obs) // T_win

    out, losses, times = [], [], []
    for n in range(n_starts):
        
        print('\n')
        print(f'optimizing window number {n+1} / {n_starts}')
        print('\n')

        idx = np.where( np.logical_and((n+1)*T_win > T_obs, T_obs >= n*T_win))[0]
        assert len(idx) > 0 # atm not supporting empty integration window

        opt_res = optim_initial_state(
            gen,
            T_rollouts=[T_win],
            T_obs=[T_obs[idx] - n*T_win],
            N=N,
            n_chunks=1,
            optimizer_pars=optimizer_pars,
            x_inits=[x_init],
            targets=[y[idx]],
            grndtrths=None,
            loss_masks=[m[idx]])

        x_sols, loss_vals, time_vals, _ = opt_res
        
        out.append(sortL96intoChannels(x_sols[0],J=J))
        losses.append(loss_vals)
        times.append(time_vals)

        priors = [SimplePrior(K=K, J=J, loc=as_tensor(out[-1][n]), scale=1.) for n in range(N)]
        
        x_init = out[-1].copy()
    
    return np.stack(out, axis=0), losses, times


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
    #assert J == 0

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
