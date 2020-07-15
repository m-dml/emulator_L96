from .util import sortL96fromChannels, sortL96intoChannels, predictor_corrector, init_torch_device
from .dataset import Dataset, DatasetRelPred, DatasetRelPredPast
from .networks import named_network
from L96_base import f1, f2

import matplotlib.pyplot as plt
import numpy as np
import torch

device, dtype = init_torch_device(), torch.float32

def solve_from_init(K, J, T_burnin, T_, dt, F, h, b, c, data, dilation=2, norm_mean=0., norm_std=1.):

    assert int(dilation) == dilation

    dX_dt = np.empty(K*(J+1), dtype=data.dtype)
    times = np.arange(0, (T_+1)*dt, dt/dilation)

    if J > 0:
        def fun(t, x):
            return f2(x, F, h, b, c, dX_dt, K, J)
    else:
        def fun(t, x):
            return f1(x, F, dX_dt, K)

    out2 = predictor_corrector(fun=fun, y0=data[T_burnin], times=times, alpha=0.5)[::dilation,:]
    out2 = sortL96fromChannels((sortL96intoChannels(out2,J) - norm_mean) / norm_std)

    return out2


def load_model_from_exp_conf(res_dir, conf):
    
    net_kwargs = {
            'filters': conf['filters'],
            'kernel_sizes': conf['kernel_sizes'],
            'filters_ks1_init': conf['filters_ks1_init'],
            'filters_ks1_inter': conf['filters_ks1_inter'],
            'filters_ks1_final': conf['filters_ks1_final'],
            'additiveResShortcuts' : conf['additiveResShortcuts'],
            'direct_shortcut' : conf['direct_shortcut']
    }

    model, model_forward = named_network(model_name=conf['model_name'], 
                                         n_input_channels=conf['J']+1, 
                                         n_output_channels=conf['J']+1,
                                         seq_length=conf['seq_length'],
                                         **net_kwargs)

    test_input = np.random.normal(size=(10, conf['seq_length']*(conf['J']+1), conf['K']))
    print(f'model output shape to test input of shape {test_input.shape}', 
          model.forward(torch.as_tensor(test_input, device=device, dtype=dtype)).shape)

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
    dtype = torch.float32
    
    if prediction_task == 'update_with_past':

        assert isinstance(dg_train, DatasetRelPredPast)
        std_out = torch.as_tensor(dg_train.std_out, device='cpu', dtype=dtype)
        mean_out = torch.as_tensor(dg_train.mean_out, device='cpu', dtype=dtype)

        def model_simulate(y0, dy0, T):
            x = np.empty((T+1, *y0.shape[1:]))
            x[0] = y0.copy()
            xx = torch.as_tensor(x[0], device='cpu', dtype=dtype)
            dx = torch.as_tensor(dy0.copy(), device='cpu', dtype=dtype)
            for i in range(1,T+1):
                xxo = xx * 1.
                xx = std_out * model_forward(torch.cat((xx.reshape(1,J+1,K), dx), axis=1)) + mean_out + xx.reshape(1,J+1,-1)
                dx = xx - xxo
                x[i] = xx.detach().numpy().copy()
            return x

    elif prediction_task == 'update': 

        assert isinstance(dg_train, DatasetRelPred)
        def model_simulate(y0, dy0, T):
            x = np.empty((T+1, *y0.shape[1:]))
            x[0] = y0.copy()
            xx = torch.as_tensor(x[0], device='cpu', dtype=dtype).reshape(1,1,-1)
            for i in range(1,T+1):
                xx = model_forward(xx.reshape(1,J+1,-1)) + xx.reshape(1,J+1,-1)
                x[i] = xx.detach().numpy().copy()
            return x

    elif prediction_task == 'state': 

        assert isinstance(dg_train, Dataset)
        def model_simulate(y0, dy0, T):
            x = np.empty((T+1, *y0.shape[1:]))
            x[0] = y0.copy()
            xx = torch.as_tensor(x[0], device='cpu', dtype=dtype).reshape(1,1,-1)
            for i in range(1,T+1):
                xx = model_forward(xx.reshape(1,J+1,-1))
                x[i] = xx.detach().numpy().copy()
            return x

    return model_simulate


def plot_rollout(out, out_model, out_comparison, T_start, T_dur, K, fig=None):

    vmax = np.maximum(np.nanmax(out[np.arange(T_dur)+T_start]),
                      np.nanmax(out_model))
    vmin = np.minimum(np.nanmin(out[np.arange(T_dur)+T_start]),
                      np.nanmin(out_model.T))

    fig = plt.figure(figsize=(16,9)) if fig is None else fig
    
    plt.figure(fig.number)
    plt.subplot(2,2,1)
    plt.imshow(out[np.arange(T_dur+1)+ T_start].T, aspect='auto', vmin=vmin, vmax=vmax)
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
    plt.plot(np.sqrt(np.mean( (out_model - out[np.arange(T_dur+1)+ T_start])**2, axis=1 )), 
             'b', label='model-reconstruction vs sim')
    plt.plot(np.sqrt(np.mean( (out_model[:,:K] - out[np.arange(T_dur+1)+ T_start][:,:K])**2, axis=1 )), 
             'b--', label='model-reconstruction vs sim, slow vars only')
    try:
        plt.plot(np.sqrt(np.mean( (out_comparison[:T_dur+1] - out[np.arange(T_dur+1)+ T_start])**2, axis=1 )),
                'k', label='solver 2x temp. resol. vs sim')
        plt.plot(np.sqrt(np.mean( (out_comparison[:T_dur+1][:,:K] - out[np.arange(T_dur+1)+ T_start][:,:K])**2, axis=1 )),
                'k--', label='solver 2x temp. resol. vs sim, slow vars only')
    except:
        pass
    plt.title('missmatch over time')
    plt.xlabel('time')
    plt.ylabel('RMSE (on z-scored data)')
    plt.legend()
    #plt.show()

    return fig