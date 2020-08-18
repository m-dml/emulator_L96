import numpy as np
import torch 

import os
import time

from L96_emulator.util import dtype, dtype_np, device
from L96_emulator.util import sortL96fromChannels, sortL96intoChannels
from L96_emulator.util import predictor_corrector, rk4_default, get_data

from L96_emulator.run import setup, sel_dataset_class

from L96_emulator.eval import load_model_from_exp_conf, get_rollout_fun, optim_initial_state, named_network

import L96sim
from L96sim.L96_base import f1, f2, pf2


def solve_initstate_fullyObs(system_pars, model_pars, optimizer_pars, setup_pars, res_dir, data_dir):

    K,J,T,dt,N_trials=system_pars['K'],system_pars['J'],system_pars['T'],system_pars['dt'],system_pars['N_trials']
    spin_up_time,train_frac,normalize_data=system_pars['spin_up_time'],system_pars['train_frac'],system_pars['normalize_data']
    F,h,b,c=system_pars['F'],system_pars['h'],system_pars['b'],system_pars['c']
    
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


    # ## L-BFGS, solve across full rollout time recursively, initialize from  explicit backward solution

    print('\n')
    print('L-BFGS, solve across full rollout time recursively, initialize from explicit backward solution')
    print('\n')

    T_rollouts = np.arange(1, n_chunks+1) * (T_rollout//n_chunks)
    grndtrths = [out[n_starts+T_rollout-(j+1)*(T_rollout//n_chunks)] for j in range(n_chunks)]
    x_inits = [None for z in range(n_chunks)]

    times_eb = dt * np.linspace(0, T_rollout//n_chunks, back_solve_dt_fac * (T_rollout//n_chunks)+1)
    def explicit_backsolve(x_init):
        x_init = sortL96fromChannels(x_init)
        x_sols = np.zeros_like(x_init)
        #x_init = torch.as_tensor(x_init, dtype=dtype, device=device)
        print('x_init.shape', x_init.shape)
        print('x_init', x_init)
        for i__ in range(x_init.shape[0]):
            out2 = rk4_default(fun=fun_eb, y0=x_init[i__], times=times_eb)
            x_sols[i__] = out2[-1].copy()#.detach().cpu().numpy().copy()
        return sortL96intoChannels(x_sols,J=J)

    x_inits[0] = explicit_backsolve(sortL96intoChannels(out[n_starts+T_rollout],J=J)).copy()
    targets = [out[n_starts+T_rollout].copy() for i in range(n_chunks)]

    res = optim_initial_state(
          model_forwarder, K, J, N,
          n_steps, optimizer_pars,
          x_inits, targets, grndtrths,
          out, n_starts, T_rollouts, n_chunks,
          f_init=explicit_backsolve)

    x_sols_LBFGS_recurse_chunks, loss_vals_LBFGS_recurse_chunks, time_vals_LBFGS_recurse_chunks, state_mses_LBFGS_recurse_chunks = res


    # ## L-BFGS, split rollout time into chunks, solve sequentially from end to beginning

    print('\n')
    print('L-BFGS, split rollout time into chunks, solve sequentially from end to beginning')
    print('\n')

    T_rollouts = np.ones(n_chunks, dtype=np.int) * (T_rollout//n_chunks)
    x_inits, targets = [None for i in range(n_chunks)], [None for i in range(n_chunks)]
    x_inits[0] = sortL96intoChannels(np.atleast_2d(out[n_starts+T_rollout].copy()),J=J)
    targets[0] = out[n_starts+T_rollout].copy()

    res = optim_initial_state(
          model_forwarder, K, J, N,
          n_steps, optimizer_pars,
          x_inits, targets, grndtrths,
          out, n_starts, T_rollouts, n_chunks)

    x_sols_LBFGS_chunks, loss_vals_LBFGS_chunks, time_vals_LBFGS_chunks, state_mses_LBFGS_chunks = res

    # ## L-BFGS, solve across full rollout time in one go, initialize from chunked approach

    print('\n')
    print('L-BFGS, solve across full rollout time in one go, initialize from chunked approach')
    print('\n')

    T_rollouts = np.arange(1, n_chunks+1) * (T_rollout//n_chunks)
    x_inits = [sortL96intoChannels(z,J=J).copy() for z in x_sols_LBFGS_chunks] 
    targets = [out[n_starts+T_rollout].copy() for i in range(n_chunks)]

    res = optim_initial_state(
          model_forwarder, K, J, N,
          n_steps, optimizer_pars,
          x_inits, targets, grndtrths,
          out, n_starts, T_rollouts, n_chunks)

    x_sols_LBFGS_full_chunks, loss_vals_LBFGS_full_chunks, time_vals_LBFGS_full_chunks, state_mses_LBFGS_full_chunks = res


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
          model_forwarder, K, J, N,
          n_steps, optimizer_pars,
          x_inits, targets, grndtrths,
          out, n_starts, T_rollouts, n_chunks)

    x_sols_LBFGS_full_backsolve, loss_vals_LBFGS_full_backsolve, time_vals_LBFGS_full_backsolve, state_mses_LBFGS_full_backsolve = res


    # ## L-BFGS, solve across full rollout time in one go
    # - warning, this can be excruciatingly slow and hard to converge !

    print('\n')
    print('L-BFGS, solve across full rollout time in one go')
    print('\n')

    x_init = sortL96intoChannels(out[n_starts+T_rollout], J=J)
    x_inits = [x_init.copy() for j in range(n_chunks)]
    res = optim_initial_state(
          model_forwarder, K, J, N,
          n_steps, optimizer_pars,
          x_inits, targets, grndtrths,
          out, n_starts, T_rollouts, n_chunks)

    x_sols_LBFGS_full_persistence, loss_vals_LBFGS_full_persistence, time_vals_LBFGS_full_persistence, state_mses_LBFGS_full_persistence = res


    # ## plot and compare results

    print('\n')
    print('done, storing results')
    print('\n')

    initial_states = [out[n_starts+j*T_rollout//n_chunks] for j in range(n_chunks)]
    initial_states = np.stack([sortL96intoChannels(z,J=J) for z in initial_states])

    model_forwarder_str = args['model_forwarder']
    optimizer_str = optimizer_pars['optimizer']
    np.save(res_dir + f'results/data_assimilation/fullyobs_initstate_tests_exp{exp_id}_{model_forwarder_str}_{optimizer_str}',
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
