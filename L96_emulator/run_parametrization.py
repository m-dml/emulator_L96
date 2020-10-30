import numpy as np
import torch
from L96_emulator.util import device, as_tensor, dtype_np, dtype
from L96_emulator.util import sortL96intoChannels, sortL96fromChannels, rk4_default
from L96_emulator.parametrization import Parametrization_lin, Parametrization_nn, Parametrized_twoLevel_L96
from L96_emulator.data_assimilation import get_model
from L96_emulator.networks import Model_forwarder_rk4default
from L96_emulator.run import sel_dataset_class, loss_function
from L96_emulator.train import train_model
from L96_emulator.likelihood import GenModel, ObsOp_identity, SimplePrior
from L96sim.L96_base import f1, f2, pf2
from configargparse import ArgParser
import subprocess

import os
def mkdir_from_path(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def run_exp_parametrization(exp_id, datadir, res_dir,
            parametrization, n_hiddens,
            K, J, T, dt, spin_up_time, l96_F, l96_h, l96_b, l96_c, train_frac, validation_frac, offset,
            model_exp_id, model_forwarder, loss_fun, batch_size, eval_every,
            lr, lr_min, lr_decay, weight_decay, max_epochs, max_patience, max_lr_patience):

    fetch_commit = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    commit_id = fetch_commit.communicate()[0].strip().decode("utf-8")
    fetch_commit.kill()

    # load model
    model_pars = {
        'exp_id' : model_exp_id,
        'model_forwarder' : model_forwarder,
        'K_net' : K,
        'J_net' : 0,
        'dt_net' : dt
    }

    if model_forwarder == 'rk4_default':
        model_forwarder = rk4_default
    elif model_forwarder == 'predictor_corrector':
        model_forwarder = predictor_corrector

    model, model_forwarder, args = get_model(model_pars, res_dir=res_dir, exp_dir='')
    
    # instantiate parametrizations
    if parametrization == 'linear':
        param_train = Parametrization_lin(a=as_tensor(np.array([-0.75])), b=as_tensor(np.array([-0.4])))    
        param_train = Parametrization_lin(a=as_tensor(np.array([-0.75])), b=as_tensor(np.array([-0.4])))    
    elif parametrization == 'nn':
        param_train = Parametrization_nn(n_hiddens=n_hiddens)    
    else:
        raise NotImplementedError()
    for p in model.parameters():
        p.requires_grad = False
    model_parametrized = Parametrized_twoLevel_L96(emulator=model, parametrization=param_train)
    model_forwarder_parametrized = Model_forwarder_rk4default(model=model_parametrized, dt=dt)

    print('torch.nn.Parameters of parametrization require grad: ')
    for p in model_forwarder_parametrized.model.param.parameters():
        print(p.requires_grad)
    print('torch.nn.Parameters of emulator require grad: ')
    for p in model_forwarder_parametrized.model.emulator.parameters():
        print(p.requires_grad)

    if len(offset)>1: # multi-step predictions
        print('multi-step predictions')
        gm = GenModel(model_forwarder=model_forwarder_parametrized, 
                      model_observer=ObsOp_identity(), 
                      prior=SimplePrior(J=0, K=K))
        class MultiStepForwarder(torch.nn.Module):
            def __init__(self, model, offset):
                super(MultiStepForwarder, self).__init__()
                self.model = model
                self.offset = offset
            def forward(self, x):
                return torch.stack(gm._forward(x=x, T_obs=self.offset), dim=1)
            
        model_forwarder_parametrized = MultiStepForwarder(model=gm, offset=np.asarray(offset))
        print('model forwarder', model_forwarder_parametrized)

    if parametrization == 'linear':
        print('initialized a', model_parametrized.param.a)
        print('initialized b', model_parametrized.param.b)    
    elif parametrization == 'nn':
        print('initialized first-layer weights', model_parametrized.param.layers[0].weight)
    
    # ground-truth two-level L96 model (based on Numba implementation):
    dX_dt = np.empty(K*(J+1), dtype=dtype_np)
    if J > 0:
        def fun(t, x):
            return f2(x, l96_F, l96_h, l96_b, l96_c, dX_dt, K, J)
    else:
        def fun(t, x):
            return f1(x, l96_F, dX_dt, K)
    class Torch_solver(torch.nn.Module):
        # numerical solver (from numpy/numba/Julia)
        def __init__(self, fun):
            self.fun = fun
        def forward(self, x):
            x = sortL96fromChannels(x.detach().cpu().numpy()).flatten()
            return as_tensor(sortL96intoChannels(np.atleast_2d(self.fun(0., x)), J=J))
    model_forwarder_np = Model_forwarder_rk4default(Torch_solver(fun), dt=dt)


    # create some training data from the true two-level L96
    X_init = 0.5 + np.random.randn(1,K*(J+1)) * 1.0
    X_init = l96_F * X_init.astype(dtype=dtype_np) / np.maximum(J,50)

    def model_simulate(y0, dy0, n_steps):
        x = np.empty((n_steps+1, *y0.shape[1:]), dtype=dtype_np)
        x[0] = y0.copy()
        xx = as_tensor(x[0]).reshape(1,1,-1)
        for i in range(1,n_steps+1):
            xx = model_forwarder_np(xx.reshape(1,J+1,-1))
            x[i] = xx.detach().cpu().numpy().copy()
        return x

    T_dur = int(T/dt)
    spin_up = int(spin_up_time/dt)
    print('simulating high-res (two-level L96) data')
    data_full = model_simulate(y0=sortL96intoChannels(X_init,J=J), dy0=None, n_steps=T_dur+spin_up)
    print('full data shape: ', data_full.shape)

    # two-level simulates for fast and slow variables, we only take the slow ones for training !
    data = data_full[:,0,:] 
    print('training data shape: ', data_full.shape)

    train_frac, validation_frac = 0.8,  0.1    
    DatasetClass = sel_dataset_class(prediction_task='state', N_trials=1, local=False, offset=offset)
    print('dataset class', DatasetClass)
    print('len(offset)', len(offset))
    assert train_frac + validation_frac <= 1.
    
    dg_dict = {
        'data' : data,
        'J' : 0,
        'offset' : offset[0] if len(offset)==1 else offset,
        'normalize' : False
    }

    dg_train = DatasetClass(start=spin_up, end=spin_up+int(np.floor(T_dur*train_frac)), **dg_dict)
    train_loader = torch.utils.data.DataLoader(
        dg_train, batch_size=batch_size, drop_last=True, num_workers=0
    )
    dg_val   = DatasetClass(start=int(np.ceil(T_dur*train_frac)), 
                            end=int(np.ceil(T_dur*(train_frac+validation_frac))),
                            **dg_dict)
    validation_loader = torch.utils.data.DataLoader(
        dg_val, batch_size=batch_size, drop_last=False, num_workers=0
    )

    loss_fun = loss_function(loss_fun=loss_fun, extra_args={})
    print('starting optimization of parametrization')
    training_outputs = train_model(
        model=model_forwarder_parametrized,
        train_loader=train_loader, 
        validation_loader=validation_loader, 
        device=device, 
        model_forward=model_forwarder_parametrized, 
        loss_fun=loss_fun,
        lr=lr,
        lr_min=lr_min, 
        lr_decay=lr_decay, 
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        max_patience=max_patience, 
        max_lr_patience=max_lr_patience, 
        eval_every=eval_every
    )

    if parametrization == 'linear':
        print('learned a', model_parametrized.param.a)
        print('learned b', model_parametrized.param.b)
    elif parametrization == 'nn':
        print('initialized first-layer weights', model_parametrized.param.layers[0].weight)

    save_dir = 'results/parametrization/' + exp_id + '/'
    mkdir_from_path(res_dir + save_dir)

    open(res_dir + save_dir + commit_id + '.txt', 'w')
    
    state_dict = param_train.state_dict()
    for key, value in state_dict.items():
        state_dict[key] = value.detach().cpu().numpy()

    np.save(res_dir + save_dir + 'out', 
            {
                'data_full' : data_full,
                'X_init' : data_full[-1].reshape(1,-1),
                'param_train_state_dict' : state_dict
            })
    print('done')


def setup_parametrization(conf_exp=None):

    p = ArgParser()
    p.add_argument('-c', '--conf-exp', is_config_file=True, help='config file path', default=conf_exp)
    p.add_argument('--exp_id', type=str, required=True, help='experiment id')
    p.add_argument('--datadir', type=str, required=True, help='path to data')
    p.add_argument('--res_dir', type=str, required=True, help='path to results')

    p.add_argument('--K', type=int, required=True, help='number of slow variables (grid cells)')
    p.add_argument('--J', type=int, required=True, help='number of fast variables (vertical levels)')
    p.add_argument('--T', type=int, required=True, help='length of simulation data (in time units [s])')
    p.add_argument('--dt', type=float, required=True, help='simulation step length (in time units [s])')
    p.add_argument('--spin_up_time', type=float, default=5., help='spin-up time for simulation in [s]')
    p.add_argument('--train_frac', type=float, default=0.8, help='fraction of data data for training')
    p.add_argument('--validation_frac', type=float, default=0.1, help='fraction of data for validation')
    p.add_argument('--offset', type=int, nargs='+', default=[1], help='time offset for prediction (can be list)')

    p.add_argument('--l96_F', type=float, default=10., help='Lorenz-96 parameter F')
    p.add_argument('--l96_h', type=float, default=1., help='Lorenz-96 parameter h')
    p.add_argument('--l96_b', type=float, default=10., help='Lorenz-96 parameter b')
    p.add_argument('--l96_c', type=float, default=10., help='Lorenz-96 parameter c')

    p.add_argument('--model_exp_id', type=int, required=True, help='exp_id for emulator-training experiment')
    p.add_argument('--model_forwarder', type=str, default='rk4_default', help='string for model forwarder (e.g. RK4)')
    p.add_argument('--parametrization', type=str, default='linear', help='string specifying parametrization model')
    p.add_argument('--n_hiddens', type=int, nargs='+',  default=[32,32], help='string specifying layers of NN parametrization')

    p.add_argument('--loss_fun', type=str, default='mse', help='loss function for model training')    
    p.add_argument('--batch_size', type=int, default=32, help='batch-size')
    p.add_argument('--max_epochs', type=int, default=20, help='epochs')
    p.add_argument('--max_patience', type=int, default=None, help='patience for early stopping')
    p.add_argument('--eval_every', type=int, default=None, help='frequency for checking convergence (in minibatches)')
    p.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    p.add_argument('--lr_min', type=float, default=1e-6, help='minimal learning rate after which stop reducing')
    p.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay factor')
    p.add_argument('--max_lr_patience', type=int, default=None, help='patience per learning rate plateau')
    p.add_argument('--weight_decay', type=float, default=0., help='weight decay (L2 norm)')

    args = p.parse_args() if conf_exp is None else p.parse_args(args=[])
    return vars(args)