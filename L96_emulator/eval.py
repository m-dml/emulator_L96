from L96_emulator.util import sortL96fromChannels, sortL96intoChannels, device, dtype, as_tensor
from L96_emulator.dataset import Dataset, DatasetRelPred, DatasetRelPredPast
from L96_emulator.networks import named_network

import matplotlib.pyplot as plt
import numpy as np
import torch
import time


class Rollout(torch.nn.Module):

    def __init__(self, model_forwarder, prediction_task, K, J, N, T=1, x_init=None):

        super(Rollout, self).__init__()

        self.model_forwarder = model_forwarder
        self.prediction_task = prediction_task

        self.T = T

        x_init = np.random.normal(size=(N,K*(J+1))) if x_init is None else x_init
        assert x_init.ndim in [2,3]
        self.X = torch.nn.Parameter(as_tensor(x_init))

    def forward(self):

        if self.prediction_task == 'state':
            x = self.X
            for t in range(self.T):
                x = self.model_forwarder.forward(x)
            return x
        else:
            raise NotImplementedError


def load_model_from_exp_conf(res_dir, conf):
    
    net_kwargs = {
            'filters': conf['filters'],
            'kernel_sizes': conf['kernel_sizes'],
            'filters_ks1_init': conf['filters_ks1_init'],
            'filters_ks1_inter': conf['filters_ks1_inter'],
            'filters_ks1_final': conf['filters_ks1_final'],
            'additiveResShortcuts' : conf['additiveResShortcuts'],
            'direct_shortcut' : conf['direct_shortcut'],
            'dropout_rate' : conf['dropout_rate'],
            'layerNorm' : conf['layerNorm'],
            'init_net' : conf['init_net'], 
            'K_net' : conf['K_net'], 
            'J_net' : conf['J_net'], 
            'dt_net' : conf['dt_net'], 
            'alpha_net' : conf['alpha_net'],
            'model_forwarder' : conf['model_forwarder']
    }

    model, model_forward = named_network(model_name=conf['model_name'], 
                                         n_input_channels=conf['J']+1, 
                                         n_output_channels=conf['J']+1,
                                         seq_length=conf['seq_length'],
                                         **net_kwargs)

    test_input = np.random.normal(size=(10, conf['seq_length']*(conf['J']+1), conf['K']))
    print(f'model output shape to test input of shape {test_input.shape}', 
          model.forward(as_tensor(test_input)).shape)

    print('total #parameters: ', np.sum([np.prod(item.shape) for item in model.state_dict().values()]))

    lead_time, exp_id = conf['lead_time'], conf['exp_id']
    save_dir = res_dir + 'models/' + exp_id + '/'
    model_fn = f'{exp_id}_dt{lead_time}.pt'
    model.load_state_dict(torch.load(save_dir + model_fn, map_location=torch.device(device)))
    
    try: 
        training_outputs = np.load(save_dir + '_training_outputs' + '.npy', allow_pickle=True)[()]
    except:
        training_outputs = None
        print('WARNING: could not load diagnostical outputs from model training!')
    
    return model, model_forward, training_outputs


def get_rollout_fun(dg_train, model_forward, prediction_task):

    J, K = dg_train.data.shape[1]-1, dg_train.data.shape[2]

    if prediction_task == 'update_with_past':

        raise NotImplementedError

        assert isinstance(dg_train, DatasetRelPredPast)
        std_out = as_tensor(dg_train.std_out)
        mean_out = as_tensor(dg_train.mean_out)

        def model_simulate(y0, dy0, n_steps):
            x = np.empty((n_steps+1, *y0.shape[1:]))
            x[0] = y0.copy()
            xx = as_tensor(x[0])
            dx = as_tensor(dy0.copy())
            for i in range(1,n_steps+1):
                xxo = xx * 1.
                xx = std_out * model_forward(torch.cat((xx.reshape(1,J+1,K), dx), axis=1)) + mean_out + xx.reshape(1,J+1,-1)
                dx = xx - xxo
                x[i] = xx.detach().cpu().numpy().copy()
            return x

    elif prediction_task == 'update': 

        raise NotImplementedError

        assert isinstance(dg_train, DatasetRelPred)
        def model_simulate(y0, dy0, n_steps):
            x = np.empty((n_steps+1, *y0.shape[1:]))
            x[0] = y0.copy()
            xx = as_tensor(x[0]).reshape(1,1,-1)
            for i in range(1,n_steps+1):
                xx = model_forward(xx.reshape(1,J+1,-1)) + xx.reshape(1,J+1,-1)
                x[i] = xx.detach().cpu().numpy().copy()
            return x

    elif prediction_task == 'state': 

        assert isinstance(dg_train, Dataset)
        def model_simulate(y0, dy0, n_steps):
            x = np.empty((n_steps+1, *y0.shape[1:]))
            x[0] = y0.copy()
            xx = as_tensor(x[0]).reshape(1,1,-1)
            for i in range(1,n_steps+1):
                xx = model_forward(xx.reshape(1,J+1,-1))
                x[i] = xx.detach().cpu().numpy().copy()
            return x

    return model_simulate


def plot_rollout(out, out_model, out_comparison, n_start, n_steps, K=None, fig=None):

    vmax = np.maximum(np.nanmax(out[np.arange(n_steps)+n_start]),
                      np.nanmax(out_model))
    vmin = np.minimum(np.nanmin(out[np.arange(n_steps)+n_start]),
                      np.nanmin(out_model.T))

    fig = plt.figure(figsize=(16,9)) if fig is None else fig
    
    plt.figure(fig.number)
    plt.subplot(2,2,1)
    plt.imshow(out[np.arange(n_steps+1)+ n_start].T, aspect='auto', vmin=vmin, vmax=vmax)
    plt.xlabel('time')
    plt.ylabel('location')
    plt.title('numerical simulation')
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(out_model.T, aspect='auto', vmin=vmin, vmax=vmax)
    plt.xlabel('time')
    plt.ylabel('location')
    plt.title('model-reconstructed simulation')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.plot(np.mean( (out_model - out[np.arange(n_steps+1)+ n_start])**2, axis=1 ), 
             'b', label='model-reconstruction vs sim')
    KJ1 = out_model.shape[1]
    if not K is None:
        plt.plot(np.sum( (out_model[:,:K] - out[np.arange(n_steps+1)+ n_start][:,:K])**2, axis=1 )/KJ1, 
                 'b--', label='model-reconstruction vs sim, fraction from slow vars only')
    try:
        plt.plot(np.mean( (out_comparison[:n_steps+1] - out[np.arange(n_steps+1)+ n_start])**2, axis=1 ),
                'k', label='solver 2x temp. resol. vs sim')
        if not K is None:
            plt.plot(np.sum( (out_comparison[:n_steps+1][:,:K] - out[np.arange(n_steps+1)+ n_start][:,:K])**2, axis=1 )/KJ1,
                    'k--', label='solver 2x temp. resol. vs sim, fraction from slow vars only')
    except:
        pass
    plt.title('missmatch over time')
    plt.xlabel('time (# integration steps)')
    plt.ylabel('MSE')
    plt.legend()
    #plt.show()

    return fig