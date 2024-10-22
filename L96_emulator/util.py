import numpy as np
import torch
from L96sim.L96_base import f1, f2, pf2

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

def as_tensor(x):
    return torch.as_tensor(x, dtype=dtype, device=device)

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

def calc_jakobian_onelevelL96_tendencies(inputs, n):

    inputs_m1 = np.concatenate((inputs[-1:], inputs[:-1]))
    inputs_m2 = np.concatenate((inputs[-2:], inputs[:-2]))
    inputs_p1 = np.concatenate((inputs[1:], inputs[:1]))

    dfdx = - 1. * np.eye(n, dtype=dtype_np)
    dfdx += np.diag(inputs_m1[:-1], 1) + np.diag(inputs_m1[-1:], -n+1)
    dfdx -= np.diag(inputs_p1[:-2],-2) + np.diag(inputs_p1[-2:], n-2)
    dfdx += np.diag(inputs_p1[1:]-inputs_m2[1:],-1) + np.diag(inputs_p1[:1]-inputs_m2[:1], n-1)

    return dfdx

def calc_jakobian_rk4(inputs, calc_f, calc_J_f, dt, n):

    I = np.eye(n, dtype=dtype_np)
    
    f0 = calc_f(inputs)
    f1 = calc_f(inputs + dt/2. * f0)
    f2 = calc_f(inputs + dt/2. * f1)

    J0 = calc_J_f(inputs=inputs,          n=n)
    J1 = calc_J_f(inputs=inputs+dt/2.*f0, n=n).dot(dt/2*J0+I)
    J2 = calc_J_f(inputs=inputs+dt/2.*f1, n=n).dot(dt/2*J1+I)
    J3 = calc_J_f(inputs=inputs+dt   *f2, n=n).dot(dt*J2+I)

    J = I + dt/6. * (J0 + 2 * J1 + 2 * J2 + J3)

    return J 

def get_jacobian_torch(model, inputs, n):
    J = np.zeros((n,n), dtype=dtype_np)
    for i in range(n):
        inputs.grad = None
        L = model(inputs).flatten()[i]
        L.backward()
        J[i,:] = inputs.grad.detach().cpu().numpy()
    return J


def get_data(K, J, T, dt, N_trials=1, F=10., h=1., b=10., c=10., 
             resimulate=True, solver=rk4_default, save_sim=False, data_dir=None):

    if N_trials > 1:
        fn_data = f'out_K{K}_J{J}_T{T}_N{N_trials}_dt0_{str(dt)[2:]}'
    else:
        fn_data = f'out_K{K}_J{J}_T{T}_dt0_{str(dt)[2:]}'

    if J > 0:
        if N_trials > 1:
            def fun(t, x):
                return pf2(x, F, h, b, c, dX_dt, K, J)
        else:
            def fun(t, x):
                return f2(x, F, h, b, c, dX_dt, K, J)
    else:
        def fun(t, x):
            return f1(x, F, dX_dt, K)

    times = np.linspace(0, T, int(np.floor(T/dt)+1))
    if resimulate:
        print('simulating data')
        X_init = F * (0.5 + np.random.randn(K*(J+1),N_trials) * 1.0).astype(dtype=dtype_np) / np.maximum(J,50)
        X_init = X_init[:,0] if N_trials == 1 else X_init
        dX_dt = np.empty(X_init.shape, dtype=X_init.dtype)

        out = solver(fun=fun, y0=X_init.copy(), times=times)
        if N_trials > 1:
            out = out.transpose(2,0,1)

        # filename for data storage
        if save_sim: 
            assert not data_dir is None
            np.save(data_dir + fn_data, out.astype(dtype=dtype_np))
    else:
        print('loading data')
        out = np.load(data_dir + fn_data + '.npy')
        X_init = out[0].copy() if N_trials == 1 else out[:,0,:].copy()
        dX_dt = None

    return out, {'times' : times, 'X_init' : X_init, 'dX_dt' : dX_dt}
