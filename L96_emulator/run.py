import numpy as np
import torch
from L96_emulator.util import init_torch_device, device, as_tensor
from L96_emulator.networks import named_network
from L96_emulator.dataset import Dataset, DatasetMultiTrial, DatasetMultiTrial_shattered
from L96_emulator.dataset import DatasetMultiStep, DatasetMultiTrialMultiStep
from L96_emulator.train import train_model, loss_function
from configargparse import ArgParser
import ast
import subprocess

import os
def mkdir_from_path(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def sel_dataset_class(prediction_task, N_trials, local, offset=[1]):

    assert prediction_task in ['state']
    if N_trials==1:
        if len(offset) > 1:
            DatasetClass = DatasetMultiStep
        else:
            DatasetClass = Dataset

    elif N_trials>1:
        if local:
            if len(offset) > 1:
                raise NotImplementedError()
            else:
                DatasetClass = DatasetMultiTrial_shattered
        else:
            if len(offset) > 1:
                DatasetClass = DatasetMultiTrialMultiStep                
            else:
                DatasetClass = DatasetMultiTrial
    else:
        raise NotImplementedError()

    return DatasetClass

def run_exp(exp_id, datadir, res_dir,
            K, K_local, J, T, N_trials, dt, n_local,
            prediction_task, lead_time, seq_length, train_frac, validation_frac, spin_up_time,            
            model_name, loss_fun, weight_decay, batch_size, batch_size_eval, max_epochs, eval_every, 
            max_patience, lr, lr_min, lr_decay, max_lr_patience, only_eval, normalize_data, **net_kwargs):

    fetch_commit = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    commit_id = fetch_commit.communicate()[0].strip().decode("utf-8")
    fetch_commit.kill()

    # load data    
    fn_data = f'out_K{K}_J{J}_T{T}_N{N_trials}_dt0_{str(dt)[2:]}'
    out = np.load(datadir + fn_data + '.npy')
    print('data.shape', out.shape)
    assert (out.shape[1]-1)*dt == T

    K_local = K if K_local < 0 else K_local
    DatasetClass = sel_dataset_class(prediction_task, N_trials, local=(K_local<K))
    test_frac = 1. - (train_frac + validation_frac)
    assert test_frac >= 0.
    spin_up = int(spin_up_time/dt)
    dg_args = {'data' : out,
               'J' : J, 
               'offset' : lead_time,
               'normalize' : bool(normalize_data)}
    if DatasetClass == DatasetMultiTrial_shattered:
        dg_args['K_local'] = K_local
        dg_args['n_local'] = n_local

    dg_train = DatasetClass(start=spin_up, 
                            end=int(np.floor(T/dt*train_frac)),
                            **dg_args)
    dg_val   = DatasetClass(start=int(np.ceil(T/dt*train_frac)),
                            end=int(np.ceil(T/dt*(train_frac+validation_frac))),
                            **dg_args)

    print('N training data:', len(dg_train))
    print('N validation data:', len(dg_val))

    batch_size_eval = batch_size if batch_size_eval < 1 else batch_size_eval
    validation_loader = torch.utils.data.DataLoader(
        dg_val, batch_size=batch_size_eval, drop_last=False, num_workers=0 
    )
    train_loader = torch.utils.data.DataLoader(
        dg_train, batch_size=batch_size, drop_last=True, num_workers=0
    )


    ## define model
    print('net_kwargs', net_kwargs)
    model_fn = f'{exp_id}_dt{lead_time}.pt'

    model, model_forward = named_network(model_name=model_name, 
                                         n_input_channels=J+1, 
                                         n_output_channels=J+1,
                                         seq_length=seq_length,
                                         **net_kwargs)
    try:
        print('model.layer1.weights', model.layer1.weight.shape)
    except:
        pass
    if K_local < K:
        test_input = np.random.normal(size=(10, seq_length*(J+1), K_local + 3*n_local))
    else:
        test_input = np.random.normal(size=(10, seq_length*(J+1), K))
    print(f'model output shape to test input of shape {test_input.shape}', 
          model_forward(as_tensor(test_input)).shape)
    print('total #parameters: ', np.sum([np.prod(item.shape) for item in model.state_dict().values()]))

    ## train model
    save_dir = res_dir + 'models/' + exp_id + '/'
    if only_eval:
        print('loading model from disk')
        model.load_state_dict(torch.load(save_dir + model_fn, map_location=torch.device(device)))

    else: # actually train

        mkdir_from_path(save_dir)
        print('saving model state_dict to ' + save_dir + model_fn)
        open(save_dir + commit_id + '.txt', 'w')

        output_fn = '_training_outputs'
        extra_args={}
        if loss_fun=='local_mse':
            extra_args = {
                'n_local' : n_local,
                'pad_local' : (2,2) if J==1 else (2,1) # relevant local area for L96 diff.eq. 
            }
        print('loss_fun', loss_fun)
        loss_fun = loss_function(loss_fun, extra_args=extra_args)
        print('loss_fun', loss_fun)
        training_outputs = train_model(
            model, train_loader, validation_loader, device, model_forward, loss_fun=loss_fun,
            weight_decay=weight_decay, max_epochs=max_epochs, max_patience=max_patience, 
            lr=lr, lr_min=lr_min, lr_decay=lr_decay, max_lr_patience=max_lr_patience,
            eval_every=eval_every, verbose=True, save_dir=save_dir, model_fn=model_fn, output_fn=output_fn
        )
        print('saving full model to ' + save_dir+model_fn[:-3] + '_full_model.pt')
        torch.save(model, save_dir+model_fn[:-3] + '_full_model.pt')

    # evaluate model

    # tbd

def setup(conf_exp=None):
    p = ArgParser()
    p.add_argument('-c', '--conf-exp', is_config_file=True, help='config file path', default=conf_exp)
    p.add_argument('--exp_id', type=str, required=True, help='experiment id')
    p.add_argument('--datadir', type=str, required=True, help='path to data')
    p.add_argument('--res_dir', type=str, required=True, help='path to results')
    #p.add_argument('--mmap_mode', type=str, default='r', help='memmap data read mode')    
    p.add_argument('--only_eval', type=bool, default=False, help='if to evaulate saved model (=False for training)')

    p.add_argument('--K', type=int, required=True, help='number of slow variables (grid cells)')
    p.add_argument('--K_local', type=int, default=-1, help='number of slow variables (grid cells) in local training region')
    p.add_argument('--n_local', type=int, default=0, help='number of local training regions needed for single update step')
    p.add_argument('--J', type=int, required=True, help='number of fast variables (vertical levels)')
    p.add_argument('--T', type=int, required=True, help='length of simulation data (in time units [s])')
    p.add_argument('--dt', type=float, required=True, help='simulation step length (in time units [s])')
    p.add_argument('--N_trials', type=int, default=1, help='number of random starting points for solver')

    p.add_argument('--prediction_task', type=str, required=True, help='string specifying prediction task')
    p.add_argument('--lead_time', type=int, required=True, help='forecast lead time (in time steps)')
    p.add_argument('--seq_length', type=int, default=1, help='length of input state sequence to network')
    p.add_argument('--train_frac', type=float, default=0.8, help='fraction of data data for training')
    p.add_argument('--validation_frac', type=float, default=0.1, help='fraction of data for validation')
    p.add_argument('--spin_up_time', type=float, default=5., help='spin-up time for simulation in [s]')
    p.add_argument('--normalize_data', type=int, default=1, help='bool specifying if to z-score data')

    #p.add_argument('--past_times', type=int, nargs='+', default=[], help='additional time points as input')
    #p.add_argument('--past_times_own_axis', type=bool, default=False, help='if additional input times are on own axis')

    p.add_argument('--loss_fun', type=str, default='mse', help='loss function for model training')
    p.add_argument('--batch_size', type=int, default=32, help='batch-size')
    p.add_argument('--batch_size_eval', type=int, default=-1, help='batch-size for evaluation dataset')
    p.add_argument('--max_epochs', type=int, default=2000, help='epochs')
    p.add_argument('--max_patience', type=int, default=None, help='patience for early stopping')
    p.add_argument('--eval_every', type=int, default=None, help='frequency for checking convergence (in minibatches)')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    p.add_argument('--lr_min', type=float, default=1e-6, help='minimal learning rate after which stop reducing')
    p.add_argument('--lr_decay', type=float, default=1., help='learning rate decay factor')
    p.add_argument('--max_lr_patience', type=int, default=None, help='patience per learning rate plateau')
        
    p.add_argument('--model_name', type=str, required=True, help='designator for neural network')
    p.add_argument('--padding_mode', type=str, default='circular', help='designator for padding mode')
    p.add_argument('--filters', type=int, nargs='+', required=True, help='filter count per layer or block')
    p.add_argument('--kernel_sizes', type=int, nargs='+', required=True, help='kernel sizes per layer or block')
    p.add_argument('--filters_ks1_init', type=int, nargs='+', required=False, help='initial 1x1 convs for network')
    p.add_argument('--filters_ks1_inter', type=int, nargs='+', required=False, help='intermediate 1x1 convs for network')
    p.add_argument('--filters_ks1_final', type=int, nargs='+', required=False, help='final 1x1 convs for network')
    p.add_argument('--additiveResShortcuts', default=None, help='boolean or None, if ResNet shortcuts are additive')
    p.add_argument('--direct_shortcut', type=bool, default=False, help='if model has direct input-output residual connection')
    p.add_argument('--weight_decay', type=float, default=0., help='weight decay (L2 norm)')
    p.add_argument('--dropout_rate', type=float, default=0., help='Dropout')
    p.add_argument('--layerNorm', type=str, default='BN', help='normalization layer for some network architectures')
    p.add_argument('--init_net', type=str, default='rand', help='string specfifying weight initialization for some models')
    p.add_argument('--K_net', type=int, default=1, help='number of slow variables (grid cells) for some models')
    p.add_argument('--J_net', type=int, default=0, help='number of fast variables (vertical levels) for some models')
    p.add_argument('--F_net', type=float, default=10., help='number of fast variables (vertical levels) for some models')
    p.add_argument('--model_forwarder', type=str, default='predictor_corrector', help='numerical solver scheme for model')    
    p.add_argument('--dt_net', type=float, default='0.', help='numerical integration time step for some models')
    p.add_argument('--alpha_net', type=float, default='0.5', help='predictor-corrector parameter for some models')

    args = p.parse_args() if conf_exp is None else p.parse_args(args=[])
    #args.var_dict = ast.literal_eval(args.var_dict)
    return vars(args)