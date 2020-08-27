import numpy as np
from L96_emulator.data_assimilation import solve_initstate
from L96_emulator.likelihood import ObsOp_subsampleGaussian, ObsOp_identity
from configargparse import ArgParser

import os
def mkdir_from_path(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def run_exp_DA(exp_id, datadir, res_dir,
            K, J, T, N_trials, dt, back_solve_dt_fac, spin_up_time, 
            l96_F, l96_h, l96_b, l96_c, obs_operator, obs_operator_r, obs_operator_sig2, 
            T_rollout, T_pred, n_chunks, n_chunks_recursive, n_starts,
            model_exp_id, model_forwarder, 
            optimizer, n_steps, lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size,
            if_LBFGS_chunks, if_LBFGS_full_chunks, if_backsolve, 
            if_LBFGS_full_backsolve, if_LBFGS_full_persistence, if_LBFGS_recurse_chunks):

    fn = 'results/data_assimilation/' + exp_id

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

    solve_initstate(system_pars=system_pars,
                    model_pars=model_pars,
                    optimizer_pars=optimizer_pars,
                    setup_pars=setup_pars,
                    optimiziation_schemes=optimiziation_schemes,
                    res_dir=res_dir,
                    data_dir=datadir,
                    fn=fn)

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
    p.add_argument('--if_LBFGS_chunks', type=bool, default=False, help='if to include chunks optimization scheme')
    p.add_argument('--if_LBFGS_full_chunks', type=bool, default=False, help='if to include full chunks optimization')
    p.add_argument('--if_backsolve', type=bool, default=False, help='if to include backsolve optimization scheme')
    p.add_argument('--if_LBFGS_full_backsolve', type=bool, default=False, help='if to include full backsolve optimization')
    p.add_argument('--if_LBFGS_full_persistence', type=bool, default=False, help='if to include persistence optimization')
    p.add_argument('--if_LBFGS_recurse_chunks', type=bool, default=False, help='if to include recursive optimization scheme')

    args = p.parse_args() if conf_exp is None else p.parse_args(args=[])
    return vars(args)