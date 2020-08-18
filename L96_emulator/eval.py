from .util import sortL96fromChannels, sortL96intoChannels, predictor_corrector, device, dtype
from .dataset import Dataset, DatasetRelPred, DatasetRelPredPast
from .networks import named_network

import matplotlib.pyplot as plt
import numpy as np
import torch
import time

#device, dtype = init_torch_device(), torch.float32

class Rollout(torch.nn.Module):

    def __init__(self, model_forwarder, prediction_task, K, J, N, T=1, x_init=None):

        super(Rollout, self).__init__()

        self.model_forwarder = model_forwarder
        self.prediction_task = prediction_task

        self.T = T

        x_init = np.random.normal(size=(N,K*(J+1))) if x_init is None else x_init
        assert x_init.ndim in [2,3]
        self.X = torch.nn.Parameter(torch.as_tensor(x_init, device=device, dtype=dtype))

    def forward(self):

        if self.prediction_task == 'state':
            x = self.X
            for t in range(self.T):
                x = self.model_forwarder.forward(x)
            return x
        else:
            raise NotImplementedError


def optim_initial_state(
      model_forwarder, K, J, N,
      n_steps, optimizer_pars,
      x_inits, targets, grndtrths, 
      out, n_starts, T_rollouts, n_chunks,
      f_init=None):

    x_sols = np.zeros((n_chunks, N, K*(J+1)))
    loss_vals = np.zeros(n_steps)
    time_vals = time.time() * np.ones(n_steps)
    state_mses = np.zeros(n_chunks)
    
    i_ = 0
    for j in range(n_chunks):

        print('\n')
        print(f'optimizing over chunk #{j} out of {n_chunks}')
        print('\n')

        target = sortL96intoChannels(torch.as_tensor(targets[j], dtype=dtype, device=device),J=J)

        if optimizer_pars['optimizer'] == 'LBFGS':

            for n in range(N):
                
                print('\n')
                print(f'optimizing over initial state #{n} / {N}')
                print('\n')
                
                roller_outer = Rollout(model_forwarder, prediction_task='state', K=K, J=J, 
                                       N=N, T=T_rollouts[j], 
                                       x_init=x_inits[j][n:n+1])
                optimizer = torch.optim.LBFGS(params=[roller_outer.X], 
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
                        loss = ((roller_outer.forward() - target[n])**2).mean()        
                        if torch.isnan(loss):
                            loss_vals[i_] = loss.detach().cpu().numpy()
                            i_ += 1
                            continue

                    def closure():
                        loss = ((roller_outer.forward() - target[n])**2).mean()
                        optimizer.zero_grad()
                        loss.backward()
                        return loss            
                    optimizer.step(closure)
                    loss_vals[i_+i_n] += loss.detach().cpu().numpy() / N
                    if n == N-1:
                        time_vals[i_+i_n] = time.time() - time_vals[i_+i_n]
                    print((time_vals[i_+i_n], loss_vals[i_+i_n]))
                    i_n += 1

                x_sols[j][n] = sortL96fromChannels(roller_outer.X.detach().cpu().numpy().copy())
            i_ += i_n
            
        elif optimizer_pars['optimizer'] == 'SGD':
            roller_outer = Rollout(model_forwarder, prediction_task='state', K=K, J=J, 
                                   N=N, T=T_rollouts[j], 
                                   x_init=x_inits[j])
            roller_outer.train()
            optimizer = torch.optim.SGD([roller_outer.X], lr=optimizer_pars['lr'], weight_decay=0.)

            for i in range(n_steps//n_chunks):

                with torch.no_grad():
                    loss = ((roller_outer.forward() - target)**2).mean()        
                    if torch.isnan(loss):
                        loss_vals[i_] = loss.detach().cpu().numpy()
                        i_ += 1
                        continue

                optimizer.zero_grad()
                loss = ((roller_outer.forward() - target)**2).mean()
                loss.backward()
                optimizer.step()

                loss_vals[i_] = loss.detach().cpu().numpy()
                time_vals[i_] = time.time() - time_vals[i_]
                print((time_vals[i_], loss_vals[i_]))

                i_ += 1

            x_sols[j] = sortL96fromChannels(roller_outer.X.detach().cpu().numpy().copy())
                                                      
        if j < n_chunks - 1 and targets[j+1] is None:
            targets[j+1] = x_sols[j].copy()
        state_mses[j] = ((x_sols[j] - grndtrths[j])**2).mean()

        with torch.no_grad():
            print('distance to initial value', ((x_sols[j] - grndtrths[j])**2).mean())
            print('distance to x_init', ((x_sols[j] - sortL96fromChannels(x_inits[j]))**2).mean())
            print('distance to target', ((x_sols[j] - targets[j])**2).mean())

        if j < n_chunks - 1 and x_inits[j+1] is None:
            x_inits[j+1] = sortL96intoChannels(x_sols[j], J=J).copy()
            if not f_init is None:
                x_inits[j+1] = f_init(x_inits[j+1])

    return x_sols, loss_vals, time_vals, state_mses


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

    if prediction_task == 'update_with_past':

        raise NotImplementedError

        assert isinstance(dg_train, DatasetRelPredPast)
        std_out = torch.as_tensor(dg_train.std_out, device=device, dtype=dtype)
        mean_out = torch.as_tensor(dg_train.mean_out, device=device, dtype=dtype)

        def model_simulate(y0, dy0, n_steps):
            x = np.empty((n_steps+1, *y0.shape[1:]))
            x[0] = y0.copy()
            xx = torch.as_tensor(x[0], device=device, dtype=dtype)
            dx = torch.as_tensor(dy0.copy(), device=device, dtype=dtype)
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
            xx = torch.as_tensor(x[0], device=device, dtype=dtype).reshape(1,1,-1)
            for i in range(1,n_steps+1):
                xx = model_forward(xx.reshape(1,J+1,-1)) + xx.reshape(1,J+1,-1)
                x[i] = xx.detach().cpu().numpy().copy()
            return x

    elif prediction_task == 'state': 

        assert isinstance(dg_train, Dataset)
        def model_simulate(y0, dy0, n_steps):
            x = np.empty((n_steps+1, *y0.shape[1:]))
            x[0] = y0.copy()
            xx = torch.as_tensor(x[0], device=device, dtype=dtype).reshape(1,1,-1)
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