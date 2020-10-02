import numpy as np
import torch


class Parametrized_twoLevel_L96(torch.nn.Module):

    def __init__(self, emulator, parametrization):

        super(Parametrized_twoLevel_L96, self).__init__()

        self.emulator = emulator
        self.param = parametrization

    def forward(self, x): 

        return self.emulator(x) + self.param(x)
