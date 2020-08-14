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

device, dtype, dtype_np = init_torch_device(), torch.float32, np.float32

def sortL96intoChannels(x, J):

    shape = x.shape
    K = shape[-1]//(J+1)
    assert shape[-1]/(J+1) == K

    if isinstance(x, torch.Tensor):
        out = torch.cat((x[...,:K].reshape(*shape[:-1],K,1), x[...,K:].reshape(*shape[:-1], K, J)),
                        axis=-1).permute(*range(len(shape)-1),-1,-2)        
    elif isinstance(x, np.ndarray):
        out = np.concatenate((x[...,:K].reshape(*shape[:-1],K,1), x[...,K:].reshape(*shape[:-1], K, J)),
                             axis=-1).transpose(*range(len(shape)-1),-1,-2)
    return out

def sortL96fromChannels(x):

    shape = x.shape
    J, K = shape[-2]-1,  shape[-1]

    if isinstance(x, torch.Tensor):
        out = torch.cat((x[...,0,:], 
                         x[...,1:,:].permute(*range(len(shape)-2),-1,-2).reshape(*shape[:-2], K*J)), 
                        axis=-1)
    elif isinstance(x, np.ndarray):
        out = np.concatenate((x[...,0,:], 
                              x[...,1:,:].transpose(*range(len(shape)-2),-1,-2).reshape(*shape[:-2], K*J)), 
                             axis=-1)
    return out

def predictor_corrector(fun, y0, times, alpha=0.5):

    y = np.zeros((len(times), *y0.shape), dtype=y0.dtype)
    y[0] = y0.copy()
    for i in range(1,len(times)):        
        dt = times[i] - times[i-1]

        f0 = fun(times[i-1], y[i-1]).copy()
        f1 = fun(times[i],   y[i-1] + dt*f0)

        y[i] = y[i-1] + dt * (alpha*f0 + (1.-alpha)*f1)
        
    return y

def rk4_default(fun, y0, times):

    y = np.zeros((len(times), *y0.shape), dtype=y0.dtype)
    y[0] = y0.copy()
    for i in range(1,len(times)):        
        dt = times[i] - times[i-1]

        f0 = fun(times[i-1], y[i-1]).copy()
        f1 = fun(times[i-1] + dt/2., y[i-1] + dt*f0/2.).copy()
        f2 = fun(times[i-1] + dt/2., y[i-1] + dt*f1/2.).copy()
        f3 = fun(times[i],   y[i-1] + dt*f2).copy()

        y[i] = y[i-1] + dt/6. * (f0 + 2.*f1 + 2.*f2 + f3)
        
    return y