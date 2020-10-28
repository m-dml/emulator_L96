import numpy as np
import torch

class Parametrized_twoLevel_L96(torch.nn.Module):

    def __init__(self, emulator, parametrization):

        super(Parametrized_twoLevel_L96, self).__init__()

        self.emulator = emulator
        self.param = parametrization

    def forward(self, x): 

        return self.emulator(x) + self.param(x)


class Parametrization_lin(torch.nn.Module):
    
    def __init__(self, a, b):

        super(Parametrization_lin, self).__init__()

        self.a = torch.nn.Parameter(a)
        self.b = torch.nn.Parameter(b)

    def forward(self, x):
        
        return self.a * x  + self.b

class Parametrization_nn(torch.nn.Module):
    
    def __init__(self, n_hiddens, n_in=1, n_out=1):

        super(Parametrization_nn, self).__init__()

        layers = []
        n_units = n_hiddens + [n_out]
        for n in n_units:
            layers.append(
                torch.nn.Conv1d(n_in, n, kernel_size=1, bias=True)
                #torch.nn.Linear(n_in, n, bias=True)
            )
            n_in = n
        self.layers = torch.nn.ModuleList(layers)

        self.nonlinearity = torch.nn.ReLU()

    def forward(self, x):
        
        for i,layer in enumerate(self.layers):
            x = layer(x)
            x = self.nonlinearity(x) if i < len(self.layers)-1 else x

        return x
