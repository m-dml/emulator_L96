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

def predictor_corrector(fun, y0, times, alpha=0.5):

    y = np.zeros((len(times), *y0.shape))
    y[0] = y0
    for i in range(1,len(times)):        
        dt = times[i] - times[i-1]

        f0 = fun(times[i-1], y[i-1])
        f1 = fun(times[i],   y[i-1] + dt*f0)

        y[i] = y[i-1] + dt * (alpha*f0 + (1-alpha)*f1)
        
    return y