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

    if model_name == 'ResNet':

        assert seq_length == 1
        assert np.all([i == 3 for i in kwargs['kernel_sizes']])

        n_filters_ks3 = kwargs['filters']
        n_filters_ks1 = [kwargs['filters_ks1_inter'] for i in range(len(n_filters_ks3)-1)]
        n_filters_ks1 = [kwargs['filters_ks1_init']] + n_filters_ks1 + [kwargs['filters_ks1_final']]

        additiveResShortcuts = kwargs['additiveResShortcuts']
        additiveResShortcuts = None if additiveResShortcuts == 'None' else additiveResShortcuts 
        additiveResShortcuts = False if additiveResShortcuts == 'False' else additiveResShortcuts 
        additiveResShortcuts = True if additiveResShortcuts == 'True' else additiveResShortcuts 

        normLayers = {'BN' : torch.nn.BatchNorm1d,
                      'ID' : torch.nn.Identity(),
                      torch.nn.BatchNorm2d : torch.nn.BatchNorm1d,
                      torch.nn.Identity() : torch.nn.Identity()
                     }                   
        
        model = ResNet(n_filters_ks3=n_filters_ks3, 
                       n_filters_ks1=n_filters_ks1, 
                       n_channels_in=seq_length * n_input_channels, 
                       n_channels_out=n_output_channels, 
                       padding_mode='circular',
                       layerNorm=normLayers[kwargs['layerNorm']],
                       dropout=kwargs['dropout_rate'],
                       additive=additiveResShortcuts,
                       direct_shortcut=kwargs['direct_shortcut'])

        def model_forward(input):
            return model.forward(input)

    elif mode_name in ['AnalyticModel_twoLevel', 'AnalyticModel_oneLevel']:
        pass

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

def setup_conv(in_channels, out_channels, kernel_size, bias, padding_mode, stride=1):
    """
    Select between regular and circular 1D convolutional layers.
    padding_mode='circular' returns a convolution that wraps padding around the final axis.
    """
    if padding_mode=='circular':
        return PeriodicConv1D(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=kernel_size,
                      bias=bias,
                      stride=stride,
                      padding_mode=padding_mode)
    else:
        return torch.nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=(kernel_size-1)//2,
                              stride=stride,
                              bias=bias)


class ResNetBlock(torch.nn.Module):
    """A residual block to construct residual networks.
    Comprises 2 conv1D operations with optional dropout and a normalization layer.

    Parameters
    ----------
    in_channels: int
        Number of channels of input tensor.
    kernel_size: list of (int, int)
        Size of the convolutional kernel for the residual layers.
    hidden_channels: int
        Number of output channels for first residual convolution.
    out_channels: int
        Number of output channels. If not equal to in_channels, will add
        additional 1x1 convolution.
    bias: bool
        Whether to include bias parameters in the residual-layer convolutions.
    layerNorm: function
        Normalization layer.
    activation: str
        String specifying nonlinearity.
    padding_mode: str
        How to pad the data ('circular' for wrap-around padding on last axis)
    dropout: float
        Dropout rate.
    """
    def __init__(self, in_channels, kernel_size,
                 hidden_channels=None, out_channels=None, additive=None,
                 bias=True, layerNorm=torch.nn.BatchNorm1d,
                 padding_mode='circular', dropout=0.1, activation="relu"):

        super(ResNetBlock, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        hidden_channels = out_channels if hidden_channels is None else hidden_channels

        self.additive = (out_channels == in_channels) if additive is None else additive

        self.conv1 = setup_conv(in_channels=in_channels,
                                  out_channels=hidden_channels,
                                  kernel_size=kernel_size,
                                  bias=bias,
                                  padding_mode=padding_mode)

        n_out_conv2 = out_channels if self.additive else hidden_channels
        self.conv2 = setup_conv(in_channels=hidden_channels,
                                  out_channels=n_out_conv2,
                                  kernel_size=kernel_size,
                                  bias=bias,
                                  padding_mode=padding_mode)

        if layerNorm is torch.nn.BatchNorm1d:
            self.norm1 = layerNorm(num_features=hidden_channels)
            self.norm2 = layerNorm(num_features=n_out_conv2)
        elif isinstance(layerNorm, torch.nn.Identity):
            self.norm1 = self.norm2 = layerNorm
        else:
            raise NotImplementedError

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        if not self.additive:
            self.conv1x1 = torch.nn.Conv1d(in_channels=in_channels+n_out_conv2,
                              out_channels=out_channels,
                              kernel_size=1,
                              bias=bias)
            if layerNorm is torch.nn.BatchNorm1d:
                self.norm1x1 = layerNorm(num_features=out_channels)
            elif isinstance(layerNorm, torch.nn.Identity):
                self.norm1x1 = layerNorm
            self.dropout1x1 = torch.nn.Dropout(dropout)

        if activation == "relu":
            self.activation =  torch.nn.functional.relu
        elif activation == "gelu":
            self.activation =  torch.nn.functional.gelu
        else:
            raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def forward(self, x, x_mask=None, x_key_padding_mask=None):
        """Pass the input through the encoder layer.

        Parameters
        ----------
        x: tensor
            The input sequence to the encoder layer.
        x_mask: tensor
            Mask for the input sequence (optional).
        x_key_padding_mask: tensor
            Mask for the x keys per batch (optional).
        """
        if self.additive:
            z =  self.dropout1(self.activation(self.norm1(self.conv1(x))))
            x += self.dropout2(self.activation(self.norm2(self.conv2(z))))
        else:
            z = self.dropout1(self.activation(self.norm1(self.conv1(x))))
            z = self.dropout2(self.activation(self.norm2(self.conv2(z))))
            x = self.dropout1x1(self.activation(self.norm1x1(self.conv1x1(torch.cat((x,z),axis=1)))))

        return x



class ResNet(torch.nn.Module):
    
    def __init__(self, n_filters_ks3, n_filters_ks1=None, n_channels_in=1, n_channels_out=1, 
                 padding_mode='zeros', additive=None, direct_shortcut=False,
                 layerNorm=torch.nn.BatchNorm1d, dropout=0.0):
        
        kernel_size = 3

        super(ResNet, self).__init__()
        
        self.n_filters_ks1 = [ [] for i in range(len(n_filters_ks3)+1) ] if n_filters_ks1 is None else n_filters_ks1
        assert len(self.n_filters_ks1) == len(n_filters_ks3) + 1
        
        self.direct_shortcut = direct_shortcut

        n_in = n_channels_in
        self.layers3x3 = []            
        self.layers_ks1 = [ [] for i in range(len(self.n_filters_ks1))]
        n_in = n_channels_in
        for i in range(len(self.n_filters_ks1)):
            for j in range(len(self.n_filters_ks1[i])):
                n_out = self.n_filters_ks1[i][j]
                block = ResNetBlock(in_channels = n_in, 
                                    kernel_size = 1,
                                    hidden_channels=None, 
                                    out_channels= n_out,
                                    bias=True, 
                                    layerNorm=layerNorm,
                                    padding_mode='circular', 
                                    dropout=dropout, 
                                    activation="relu",
                                    additive=additive)                
                self.layers_ks1[i].append(block)
                n_in = n_out

            if i >= len(n_filters_ks3):
                break

            n_out = n_filters_ks3[i]
            layer = setup_conv(in_channels=n_in,
                               out_channels=n_out,
                               kernel_size=kernel_size,
                               bias=True,
                               padding_mode=padding_mode)
            self.layers3x3.append(layer)
            n_in = n_out
            
        self.layers3x3 = torch.nn.ModuleList(self.layers3x3)
        self.layers1x1 = sum(self.layers_ks1, [])
        self.layers1x1 = torch.nn.ModuleList(self.layers1x1)
        self.final = torch.nn.Conv1d(in_channels=n_in,
                                     out_channels=n_channels_out,
                                     kernel_size= 1)
        self.nonlinearity = torch.nn.ReLU()
    
    def forward(self, x):

        if self.direct_shortcut:
            out = x
        
        for layer in self.layers_ks1[0]:
            x = self.nonlinearity(layer(x))        
        for i, layer3x3 in enumerate(self.layers3x3):
            x = self.nonlinearity(layer3x3(x))
            for layer in self.layers_ks1[i+1]:
                x = self.nonlinearity(layer(x))

        if self.direct_shortcut:
            return self.final(x) + out
        else:
            return self.final(x)

    
class TinyNetwork(torch.nn.Module):
    
    def __init__(self, n_filters_ks3, n_filters_ks1=None, n_channels_in=1, n_channels_out=1, padding_mode='zeros'):
        
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


    def forward(self, x):

        n_channels_in = x.shape[1]
        out = x
        #assert n_channels_in//2 == n_channels_in/2
        #out = x[:, n_channels_in//2:]

        for layer in self.layers_ks1[0]:
            x = self.nonlinearity(layer(x))

        for i, layer3x3 in enumerate(self.layers3x3):
            x = self.nonlinearity(layer3x3(x))
            for layer in self.layers_ks1[i+1]:
                x = self.nonlinearity(layer(x))

        return self.final(x) + out


class AnalyticModel_oneLevel():

    def __init__(self, K, F=10., loc=1., scale=1e6):

        self.K = K
        self.nonlinearity = lambda x: x**2

        # approximate identity through x = s * (x/s + l)^2 / (2*l) + s * l
        self.scale, self.loc = scale, loc
        self.loc_grad = 2. * self.loc

        kplus1, kminus1, kminus2 = self.td_mat(K,1), self.td_mat(K,-1), self.td_mat(K,-2)

        self.W1 = np.vstack(
            (kminus1, 
             kplus1 - kminus2,
             kminus1 + kplus1 - kminus2,
             np.eye(K) / self.scale)
        )
        self.b1 = np.zeros((4*K,1))
        self.b1[3*K:,:] = self.loc * np.ones((K,1))

        self.W2 = np.hstack(
            (- np.eye(K) / 2., 
             - np.eye(K) / 2., 
               np.eye(K) / 2., 
             - self.scale / self.loc_grad * np.eye(K))
        )
        self.b2 = F * np.ones(K) -  self.W2.dot(self.b1**2)

    def td_mat(self, K, k):
        if K <= 0:
            return np.array([[1.]]) if k==0 else np.array([[]])
        ak, koff = np.abs(k), K+k if k <= 0 else k-K 
        return np.diag(np.ones(K-ak), k=k) + np.diag(np.ones(ak), k=koff)
        
    def forward(self, x):
        x_ = x.reshape(-1, 1) if x.ndim == 1 else x
        out = self.W2.dot(self.nonlinearity(self.W1.dot(x_) + self.b1)) + self.b2
        return out.flatten() if x.ndim == 1 else out


class AnalyticModel_twoLevel(AnalyticModel_oneLevel):

    def __init__(self, K, J, F=10., b=10., c=10., h=1., loc=1., scale=1e6):

        super(AnalyticModel_twoLevel, self).__init__(K=K, F=F, loc=loc, scale=scale)

        kminus1, kplus1, kplus2 = self.td_mat(J*K,-1), self.td_mat(J*K,1), self.td_mat(J*K,2)
        # block matrix: W1 = [W1 for X,    0, 
        #                        0,     W1 for Y]
        W1_Y = np.vstack((kplus1, 
                          kminus1 - kplus2,
                          kplus1 + kminus1 - kplus2,
                          np.eye(J*K) / self.scale))
        self.W1 = np.vstack((np.hstack((self.W1, np.zeros((4*K, J*K)))), 
                             np.hstack((np.zeros((4*J*K, K)), W1_Y))))

        self.b1 = np.vstack((self.b1, np.zeros((4*J*K,1))))
        self.b1[-J*K:,:] = self.loc * np.ones((J*K,1))

        eyes = c * h / J * np.eye(K).flatten()
        eyes = [eyes for i in range(J)]

        self.W2 = np.hstack((self.W2, np.zeros((K, 4*J*K))))
        # dependency of X_k on <Y_{k,J}>
        self.W2[:, -J*K:] = - self.scale / self.loc_grad * np.vstack(eyes).T.reshape(K, J*K)
        # dependency of Y_{k,j} on X_k
        W2_Y = np.zeros((J*K, 4*K))
        W2_Y[:,-K:] = self.scale / self.loc_grad * np.vstack(eyes).T.reshape(K,J*K).T
        # dependencies of Y_{k,j} on Y_{k,~j}
        W2_Y = np.hstack((W2_Y,
                          c * np.hstack(
                                  (- b * np.eye(J*K) / 2.,
                                   - b * np.eye(J*K) / 2.,
                                     b * np.eye(J*K) / 2.,
                                   - self.scale / self.loc_grad * np.eye(J*K))  
                          )
                         ))
        # block matrix: W2 = [W2 for X, `mean of Y`, 
        #                     `add X`,    W2 for Y]        
        self.W2 = np.vstack((self.W2, W2_Y))

        self.b2 = np.zeros((K*(J+1),1))
        self.b2[:K,:] = F
        self.b2 = self.b2 -  self.W2.dot(self.b1**2)
