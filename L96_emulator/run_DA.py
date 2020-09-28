import numpy as np
import torch
from L96_emulator.data_assimilation import solve_initstate, solve_4dvar, get_model
from L96_emulator.likelihood import ObsOp_subsampleGaussian, ObsOp_identity, ObsOp_rotsampleGaussian, GenModel
from L96_emulator.util import rk4_default, predictor_corrector, get_data
from L96_emulator.util import as_tensor, sortL96intoChannels, sortL96fromChannels
from configargparse import ArgParser
import subprocess

import os
def mkdir_from_path(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def run_exp_DA(exp_id, datadir, res_dir,
            K, J, T, N_trials, dt, back_solve_dt_fac, spin_up_time, 
            l96_F, l96_h, l96_b, l96_c, obs_operator, obs_operator_r, obs_operator_sig2, obs_operator_frq,
            T_rollout, T_pred, n_chunks, n_chunks_recursive, n_starts,
            model_exp_id, model_forwarder, 
            optimizer, n_steps, lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size,
            if_LBFGS_chunks, if_LBFGS_full_chunks, if_backsolve, 
            if_LBFGS_full_backsolve, if_LBFGS_full_persistence, if_LBFGS_recurse_chunks):

    fetch_commit = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    commit_id = fetch_commit.communicate()[0].strip().decode("utf-8")
    fetch_commit.kill()

    system_pars = {
        'K' : K,
        'J' : J,
        'T' : T,
        'dt' : dt,
        'back_solve_dt_fac' : back_solve_dt_fac,
        'N_trials' : N_trials,
        'spin_up_time' : spin_up_time,
        'F' : l96_F,
        'h' : l96_h,
        'b' : l96_b,
        'c' : l96_c,
    }

    if obs_operator=='ObsOp_subsampleGaussian':
        system_pars['obs_operator'] = ObsOp_subsampleGaussian
        system_pars['obs_operator_args'] = {'r' : obs_operator_r, 'sigma2' : obs_operator_sig2}
    elif obs_operator=='ObsOp_identity':
        system_pars['obs_operator'] = ObsOp_identity
        system_pars['obs_operator_args'] = {}
    elif obs_operator=='ObsOp_rotsampleGaussian':
        obs_pars['obs_operator'] = ObsOp_rotsampleGaussian
        obs_pars['obs_operator_args'] = {'frq' : obs_operator_frq, 
                                         'sigma2' : obs_operator_sig2}
    else:
        raise NotImplementedError()
        
    assert T_pred >= T_rollout

    setup_pars = {
        'n_starts' : np.asarray(np.linspace(int(system_pars['spin_up_time']/system_pars['dt']),
                                            int(system_pars['T']/system_pars['dt']) - T_pred - 1,
                                            n_starts),
                                dtype=np.int),
        'T_rollout' : T_rollout,
        'T_pred' : T_pred,
        'n_chunks' : n_chunks,
        'n_chunks_recursive' : n_chunks_recursive,
    }

    model_pars = {
        'exp_id' : model_exp_id if not model_exp_id=='None' else None,
        'model_forwarder' : model_forwarder,
        'K_net' : system_pars['K'],
        'J_net' : system_pars['J'],
        'dt_net' : system_pars['dt']
    }

    optimizer_pars = {
                  'optimizer' : optimizer,
                  'n_steps' : n_steps,
                  'lr' : lr,
                  'max_iter' : max_iter,
                  'max_eval' : None if max_eval < 0 else max_eval,
                  'tolerance_grad' : tolerance_grad,
                  'tolerance_change' : tolerance_change,
                  'history_size': history_size
    }

    optimiziation_schemes = {
        'LBFGS_chunks' : if_LBFGS_chunks,
        'LBFGS_full_chunks' : if_LBFGS_full_chunks,
        'backsolve' : if_backsolve, 
        'LBFGS_full_backsolve' : if_LBFGS_full_backsolve,
        'LBFGS_full_persistence' : if_LBFGS_full_persistence, 
        'LBFGS_recurse_chunks' : if_LBFGS_recurse_chunks
    }

    save_dir = 'results/data_assimilation/' + exp_id + '/'
    mkdir_from_path(res_dir + save_dir)

    open(res_dir + save_dir + commit_id + '.txt', 'w')
    fn = save_dir + 'res'

    solve_initstate(system_pars=system_pars,
                    model_pars=model_pars,
                    optimizer_pars=optimizer_pars,
                    setup_pars=setup_pars,
                    optimiziation_schemes=optimiziation_schemes,
                    res_dir=res_dir,
                    data_dir=datadir,
                    fn=fn)

def run_exp_4DVar(exp_id, datadir, res_dir,
            T_win, T_shift, B,
            K, J, T, N_trials, dt, spin_up_time,
            l96_F, l96_h, l96_b, l96_c, obs_operator, obs_operator_r, obs_operator_sig2, obs_operator_frq,
            model_exp_id, model_forwarder, 
            optimizer, n_steps, lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size):

    fetch_commit = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    commit_id = fetch_commit.communicate()[0].strip().decode("utf-8")
    fetch_commit.kill()

    T_shift = T_win if T_shift < 0 else T_shift # backwards compatibility to older exps with T_shift=T_win

    model_pars = {
        'exp_id' : model_exp_id,
        'model_forwarder' : model_forwarder,
        'K_net' : K,
        'J_net' : J,
        'dt_net' : dt
    }

    optimizer_pars = {
                  'optimizer' : optimizer,
                  'n_steps' : n_steps,
                  'lr' : lr,
                  'max_iter' : max_iter,
                  'max_eval' : None if max_eval < 0 else max_eval,
                  'tolerance_grad' : tolerance_grad,
                  'tolerance_change' : tolerance_change,
                  'history_size': history_size
    }

    if model_forwarder == 'rk4_default':
        model_forwarder = rk4_default
    if model_forwarder == 'predictor_corrector':
        model_forwarder = predictor_corrector

    out, datagen_dict = get_data(K=K, J=J, T=T+spin_up_time, dt=dt, N_trials=N_trials, 
                                 F=l96_F, h=l96_h, b=l96_b, c=l96_c, 
                                 resimulate=True, solver=model_forwarder,
                                 save_sim=False, data_dir='')
    out = sortL96intoChannels(out.transpose(1,0,2)[int(spin_up_time/dt):], J=J)
    print('out.shape', out.shape)

    model, model_forwarder, args = get_model(model_pars, res_dir=res_dir, exp_dir='')

    obs_pars = {}
    if obs_operator=='ObsOp_subsampleGaussian':
        obs_pars['obs_operator'] = ObsOp_subsampleGaussian
        obs_pars['obs_operator_args'] = {'r' : obs_operator_r, 'sigma2' : obs_operator_sig2}
    elif obs_operator=='ObsOp_identity':
        obs_pars['obs_operator'] = ObsOp_identity
        obs_pars['obs_operator_args'] = {}
    elif obs_operator=='ObsOp_rotsampleGaussian':
        obs_pars['obs_operator'] = ObsOp_rotsampleGaussian
        obs_pars['obs_operator_args'] = {'frq' : obs_operator_frq, 
                                         'sigma2' : obs_operator_sig2}
    else:
        raise NotImplementedError()
    model_observer = obs_pars['obs_operator'](**obs_pars['obs_operator_args'])


    prior = torch.distributions.normal.Normal(loc=torch.zeros((1,J+1,K)), 
                                              scale=B*torch.ones((1,J+1,K)))
    gen = GenModel(model_forwarder, model_observer, prior)

    y = sortL96fromChannels(gen._sample_obs(as_tensor(out))) # sets the loss masks!
    m = torch.stack(gen.masks,dim=0)

    save_dir = 'results/data_assimilation/' + exp_id + '/'
    mkdir_from_path(res_dir + save_dir)

    open(res_dir + save_dir + commit_id + '.txt', 'w')
    fn = save_dir + 'res'

    print('4D-VAR')
    x_sols, losses, times = solve_4dvar(y, m,
                                        T_obs=np.arange(y.shape[0]),
                                        T_win=T_win,
                                        T_shift=T_shift,
                                        x_init=None,
                                        model_pars=model_pars,
                                        obs_pars=obs_pars,
                                        optimizer_pars=optimizer_pars,
                                        res_dir=res_dir)

    np.save(res_dir + save_dir + 'out', 
            {'out' : out,
             'y' : y.detach().cpu().numpy(),
             'm' : m.detach().cpu().numpy(),
             'x_sols' : x_sols,
             'losses' : losses,
             'times' : times,
             'T_win' : T_win,
             'T_shift' : T_shift
            })
    print('x_sols.shape', x_sols.shape)
    print('done')

def setup_DA(conf_exp=None):
    p = ArgParser()
    p.add_argument('-c', '--conf-exp', is_config_file=True, help='config file path', default=conf_exp)
    p.add_argument('--exp_id', type=str, required=True, help='experiment id')
    p.add_argument('--datadir', type=str, required=True, help='path to data')
    p.add_argument('--res_dir', type=str, required=True, help='path to results')

    p.add_argument('--K', type=int, required=True, help='number of slow variables (grid cells)')
    p.add_argument('--J', type=int, required=True, help='number of fast variables (vertical levels)')
    p.add_argument('--T', type=int, required=True, help='length of simulation data (in time units [s])')
    p.add_argument('--dt', type=float, required=True, help='simulation step length (in time units [s])')
    p.add_argument('--N_trials', type=int, default=1, help='number of random starting points for solver')
    p.add_argument('--back_solve_dt_fac', type=int, default=100, help='step size decrease factor for backsolve')
    p.add_argument('--spin_up_time', type=float, default=5., help='spin-up time for simulation in [s]')

    p.add_argument('--l96_F', type=float, default=10., help='Lorenz-96 parameter F')
    p.add_argument('--l96_h', type=float, default=1., help='Lorenz-96 parameter h')
    p.add_argument('--l96_b', type=float, default=10., help='Lorenz-96 parameter b')
    p.add_argument('--l96_c', type=float, default=10., help='Lorenz-96 parameter c')

    p.add_argument('--obs_operator', type=str, required=True, help='string for observation operator class')
    p.add_argument('--obs_operator_r', type=float, default=0., help='fraction of unobserved state entries')
    p.add_argument('--obs_operator_sig2', type=float, default=1.0, help='variance of additive observation noise')
    p.add_argument('--obs_operator_frq', type=int, default=4, help='cycle length for rotating observation operator')

    p.add_argument('--T_rollout', type=int, required=True, help='maximum length of rollout')
    p.add_argument('--T_pred', type=int, required=True, help='prediction time')
    p.add_argument('--n_chunks', type=int, required=True, help='number of chunks for rollout (e.g. one obs per chunk)')
    p.add_argument('--n_chunks_recursive', type=int, required=True, help='finer-scale chunking, e.g. for backsolve')
    p.add_argument('--n_starts', type=int, required=True, help='number of rollout trials to include')

    p.add_argument('--model_exp_id', type=int, required=True, help='exp_id for emulator-training experiment')
    p.add_argument('--model_forwarder', type=str, default='rk4_default', help='string for model forwarder (e.g. RK4)')
    p.add_argument('--optimizer', type=str, default='LBFGS', help='string specifying numerical optimizer')
    p.add_argument('--n_steps', type=int, required=True, help='number of optimization steps')
    p.add_argument('--lr', type=float, default=1e0, help='learning rate')
    p.add_argument('--max_iter', type=int, default=1000, help='number of maximum iterations per step')
    p.add_argument('--max_eval', type=int, default=-1, help='number of maximum evaluations per step')
    p.add_argument('--tolerance_grad', type=float, default=1e-12, help='convergence criterion for gradient norm')
    p.add_argument('--tolerance_change', type=float, default=1e-12, help='convergence criterion for function change')
    p.add_argument('--history_size', type=int, default=10, help='history size for L-BFGS')
    p.add_argument('--if_LBFGS_chunks', type=int, default=0, help='if to include chunks optimization scheme')
    p.add_argument('--if_LBFGS_full_chunks', type=int, default=0, help='if to include full chunks optimization')
    p.add_argument('--if_backsolve', type=int, default=0, help='if to include backsolve optimization scheme')
    p.add_argument('--if_LBFGS_full_backsolve', type=int, default=0, help='if to include full backsolve optimization')
    p.add_argument('--if_LBFGS_full_persistence', type=int, default=0, help='if to include persistence optimization')
    p.add_argument('--if_LBFGS_recurse_chunks', type=int, default=0, help='if to include recursive optimization scheme')

    args = p.parse_args() if conf_exp is None else p.parse_args(args=[])
    return vars(args)

def setup_4DVar(conf_exp=None):
    p = ArgParser()
    p.add_argument('-c', '--conf-exp', is_config_file=True, help='config file path', default=conf_exp)
    p.add_argument('--exp_id', type=str, required=True, help='experiment id')
    p.add_argument('--datadir', type=str, required=True, help='path to data')
    p.add_argument('--res_dir', type=str, required=True, help='path to results')

    p.add_argument('--T_win', type=int, required=True, help='4D-Var integration window in steps')
    p.add_argument('--T_shift', type=int, default=-1, help='per-analysis shift of 4D-Var integration window in steps')
    p.add_argument('--B', type=float, required=True, help='scale for covariance matrix of init-state prior')

    p.add_argument('--K', type=int, required=True, help='number of slow variables (grid cells)')
    p.add_argument('--J', type=int, required=True, help='number of fast variables (vertical levels)')
    p.add_argument('--T', type=int, required=True, help='length of simulation data (in time units [s])')
    p.add_argument('--dt', type=float, required=True, help='simulation step length (in time units [s])')
    p.add_argument('--N_trials', type=int, default=1, help='number of random starting points for solver')
    p.add_argument('--spin_up_time', type=float, default=5., help='spin-up time for simulation in [s]')

    p.add_argument('--l96_F', type=float, default=10., help='Lorenz-96 parameter F')
    p.add_argument('--l96_h', type=float, default=1., help='Lorenz-96 parameter h')
    p.add_argument('--l96_b', type=float, default=10., help='Lorenz-96 parameter b')
    p.add_argument('--l96_c', type=float, default=10., help='Lorenz-96 parameter c')

    p.add_argument('--obs_operator', type=str, required=True, help='string for observation operator class')
    p.add_argument('--obs_operator_r', type=float, default=0., help='fraction of unobserved state entries')
    p.add_argument('--obs_operator_sig2', type=float, default=1.0, help='variance of additive observation noise')
    p.add_argument('--obs_operator_frq', type=int, default=4, help='cycle length for rotating observation operator')

    p.add_argument('--model_exp_id', type=int, required=True, help='exp_id for emulator-training experiment')
    p.add_argument('--model_forwarder', type=str, default='rk4_default', help='string for model forwarder (e.g. RK4)')
    p.add_argument('--optimizer', type=str, default='LBFGS', help='string specifying numerical optimizer')
    p.add_argument('--n_steps', type=int, required=True, help='number of optimization steps')
    p.add_argument('--lr', type=float, default=1e0, help='learning rate')
    p.add_argument('--max_iter', type=int, default=1000, help='number of maximum iterations per step')
    p.add_argument('--max_eval', type=int, default=-1, help='number of maximum evaluations per step')
    p.add_argument('--tolerance_grad', type=float, default=1e-12, help='convergence criterion for gradient norm')
    p.add_argument('--tolerance_change', type=float, default=1e-12, help='convergence criterion for function change')
    p.add_argument('--history_size', type=int, default=10, help='history size for L-BFGS')

    args = p.parse_args() if conf_exp is None else p.parse_args(args=[])
    return vars(args)