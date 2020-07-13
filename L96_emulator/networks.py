import torch
import torch.nn.functional as F
import numpy as np

def named_network(model_name, n_input_channels, n_output_channels, seq_length, **kwargs):

    if model_name in ['TinyNetwork', 'TinyResNet']:

        assert seq_length == 1
        assert np.all([i == 3 for i in kwargs['kernel_sizes']])

        n_filters_ks3 = kwargs['filters']
        n_filters_ks1 = [kwargs['filters_ks1_inter'] for i in range(len(n_filters_ks3)-1)]
        n_filters_ks1 = [kwargs['filters_ks1_init']] + n_filters_ks1 + [kwargs['filters_ks1_final']]

        Network = TinyNetwork if model_name == 'TinyNetwork' else TinyResNet
        model = Network(n_filters_ks3=n_filters_ks3, 
                        n_filters_ks1=n_filters_ks1, 
                        n_channels_in=seq_length * n_input_channels, 
                        n_channels_out=n_output_channels, 
                        padding_mode='circular')

        def model_forward(input):
            return model.forward(input)


    return model, model_forward


class PeriodicConv1D(torch.nn.Conv1d):
    """ Implementing 1D convolutional layer with circular padding.

    """

    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding_circ = (self.padding[0] // 2, (self.padding[0] - 1) // 2)
            return F.conv1d(F.pad(input, expanded_padding_circ, mode='circular'), 
                            self.weight, self.bias, self.stride,
                            (0,), self.dilation, self.groups)
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class TinyNetwork(torch.nn.Module):
    
    def __init__(self, n_filters_ks3, n_filters_ks1 = None, n_channels_in = 1, n_channels_out = 1, padding_mode='zeros'):
        
        kernel_size = 3

        super(TinyNetwork, self).__init__()
        
        self.n_filters_ks1 = [ [] for i in range(len(n_filters_ks3)+1) ] if n_filters_ks1 is None else n_filters_ks1
        assert len(self.n_filters_ks1) == len(n_filters_ks3) + 1

        n_in = n_channels_in
        self.layers3x3 = []            
        self.layers_ks1 = [ [] for i in range(len(self.n_filters_ks1))]
        n_in = n_channels_in
        for i in range(len(self.n_filters_ks1)):
            for j in range(len(self.n_filters_ks1[i])):
                n_out = self.n_filters_ks1[i][j]
                layer = torch.nn.Conv1d(in_channels = n_in, 
                                        out_channels = n_out, 
                                        kernel_size = 1, 
                                        padding = 0, 
                                        bias = True, 
                                        padding_mode = padding_mode)
                self.layers_ks1[i].append(layer)
                n_in = n_out

            if i >= len(n_filters_ks3):
                break

            n_out = n_filters_ks3[i]
            layer = PeriodicConv1D(in_channels = n_in, 
                                   out_channels = n_out, 
                                   kernel_size = kernel_size, 
                                   padding = kernel_size, 
                                   bias = True, 
                                   padding_mode = padding_mode)
            self.layers3x3.append(layer)
            n_in = n_out
            
        self.layers3x3 = torch.nn.ModuleList(self.layers3x3)
        #self.layers_ks1 = [torch.nn.ModuleList(layers) for layers in self.layers_ks1]
        self.layers1x1 = sum(self.layers_ks1, [])
        self.layers1x1 = torch.nn.ModuleList(self.layers1x1)
        self.final = torch.nn.Conv1d(in_channels=n_in,
                                     out_channels=n_channels_out,
                                     kernel_size= 1)
        self.nonlinearity = torch.nn.ReLU()
    
    def forward(self, x):

        for layer in self.layers_ks1[0]:
            x = self.nonlinearity(layer(x))        
        for i, layer3x3 in enumerate(self.layers3x3):
            x = self.nonlinearity(layer3x3(x))
            for layer in self.layers_ks1[i+1]:
                x = self.nonlinearity(layer(x))
                
        return self.final(x)

class TinyResNet(TinyNetwork):

#    def __init__(self, n_filters_ks3, n_filters_ks1 = None, n_channels_in = 1, n_channels_out = 1, padding_mode='zeros'):
#        super(TinyResNet, self).__init__(n_filters_ks3, n_filters_ks1, n_channels_in, n_channels_out, padding_mode)

    def forward(self, x):

        n_channels_in = x.shape[1]
        out = x
        #assert n_channels_in//2 == n_channels_in/2
        #out = x[:, n_channels_in//2:]

        for layer in self.layers_ks1[0]:
            x = self.nonlinearity(layer(x))
        #out += x # outcomment for initial residual block

        for i, layer3x3 in enumerate(self.layers3x3):
            x = self.nonlinearity(layer3x3(x))
            for layer in self.layers_ks1[i+1]:
                x = self.nonlinearity(layer(x))

        return self.final(x) + out