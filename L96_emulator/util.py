import numpy as np
import torch

def init_torch_device():
    if torch.cuda.is_available():
        print('using CUDA !')
        device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("CUDA not available")
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
    return device

def sortL96intoChannels(x, J):

    assert x.ndim == 2
    K = x.shape[1]//(J+1)
    assert x.shape[1]/(J+1) == K

    if isinstance(x, torch.Tensor):
        out = torch.cat((x[:,:K].reshape(-1,K,1), x[:,K:].reshape(-1, K, J)),
                        axis=2).permute(0,2,1)        
    elif isinstance(x, np.ndarray):
        out = np.concatenate((x[:,:K].reshape(-1,K,1), x[:,K:].reshape(-1, K, J)),
                             axis=2).transpose(0,2,1)
    return out

def sortL96fromChannels(x):

    assert x.ndim == 3
    J, K = x.shape[1]-1,  x.shape[2]

    if isinstance(x, torch.Tensor):
        out = torch.cat((x[:,0,:], x[:,1:,:].permute(0,2,1).reshape(-1, K*J)), axis=1)
    elif isinstance(x, np.ndarray):
        out = np.concatenate((x[:,0,:], x[:,1:,:].transpose(0,2,1).reshape(-1, K*J)), axis=1)
    return out

def predictor_corrector(fun, y0, times, alpha=0.5):

    y = np.zeros((len(times), *y0.shape))
    y[0] = y0
    for i in range(1,len(times)):        
        dt = times[i] - times[i-1]

        f0 = fun(times[i-1], y[i-1])
        f1 = fun(times[i],   y[i-1] + dt*f0)

        y[i] = y[i-1] + dt * (alpha*f0 + (1-alpha)*f1)
        
    return y