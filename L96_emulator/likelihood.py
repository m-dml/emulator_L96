import numpy as np
import torch 

from L96_emulator.util import as_tensor

class ObsOp_identity(torch.nn.Module):
    def __init__(self):
        super(ObsOp_identity, self).__init__()
        self.sigma = torch.tensor([1.0]) # note we have non-zero sigma for log_prob !
        self.ndistr = torch.distributions.normal.Normal(loc=0., scale=self.sigma)
        self.mask = 1.

    def _sample_mask(self, sample_shape):
        self.mask = torch.ones(size=sample_shape, dtype=torch.int)

    def forward(self, x): # deterministic part of observation operator
        """ y = f(x)  """
        assert len(x.shape) == 3 # N x J+1 x K
        return x

    def sample(self, x, m=None): # observation operator (incl. stochastic parts)
        """ sample y ~ p(y |x, m) """
        if m is None:
            self._sample_mask(sample_shape=x.shape)
            return self.forward(x)
        assert x.shape == m.shape
        return m * self.forward(x)

    def log_prob(self, y, x, m=None):
        """ log p(y|x, m)  """
        assert len(y.shape) == 3 # N x J+1 x K

        x = x.reshape(1, *x.shape) if len(x.shape)==2 else x
        assert x.shape[1:] == y.shape[1:]

        if m is None:
            return self.ndistr.log_prob(x - y).sum() # sum from iid over dims

        m = m.reshape(1, *m.shape) if len(m.shape)==2 else m
        assert m.shape[1:] == y.shape[1:]
        return (m * self.ndistr.log_prob(x - y)).sum(axis=(-2,-1)) # sum from iid over dims


class ObsOp_subsampleGaussian(ObsOp_identity):
    def __init__(self, r=0., sigma2=0.):
        super(ObsOp_subsampleGaussian, self).__init__()
        assert sigma2 >= 0.
        self.sigma2 = as_tensor(sigma2)
        self.sigma = torch.sqrt(self.sigma2)
        self.ndistr = torch.distributions.normal.Normal(loc=0., scale=self.sigma)

        assert 0. <= r <=1.
        self.r = as_tensor(r)
        self.mdistr = torch.distributions.Bernoulli(probs=1-self.r)
        self.mask = 1.

    def _sample_mask(self, sample_shape):
        self.mask = self.mdistr.sample(sample_shape=sample_shape)

    def sample(self, x, m=None): # observation operator (incl. stochastic parts)
        """ sample y ~ p(y |x, m) """
        if m is None:
            self._sample_mask(sample_shape=x.shape)
            m = self.mask
        eps = self.sigma * self.ndistr.sample(sample_shape=x.shape)
        return m * (self.forward(x) + eps)

    def log_prob(self, y, x, m=None):
        """ log p(y|x, m)  """
        assert len(y.shape) == 3 # N x J+1 x K

        x = x.reshape(1, *x.shape) if len(x.shape)==2 else x
        if m is None:
            m = self.mask
        m = m.reshape(1, *m.shape) if len(m.shape)==2 else m

        assert y.shape[1:] == m.shape[1:] and x.shape[1:] == y.shape[1:]
        return (m * self.ndistr.log_prob(x - y)).sum(axis=(-2,-1)) # sum from iid over dims


class GenModel(torch.nn.Module):

    def __init__(self, model_forwarder, model_observer, prior, 
                 T=1, x_init=None):

        super(GenModel, self).__init__()

        self.model_forwarder = model_forwarder
        self.set_rollout_len(T)

        self.model_observer = model_observer
        self.masks = [self.model_observer.mask]

        self.prior = prior        

        # variable container for e.g. maximim-likelihood estimate: 
        x_init = self.prior.sample() if x_init is None else x_init
        assert len(x_init.shape) in [2,3]
        self.set_state(x_init)

    def _forward(self, x=None, T_obs=None):

        x = self.X if x is None else x
        if len(x.shape) == 2:
            x = x.reshape(1, *x.shape)
        assert len(x.shape) == 3 # N x J+1 x K

        T_obs = [self.T-1] if T_obs is None else T_obs

        y = []
        for t in range(np.max(T_obs) + 1):
            x = self.model_forwarder.forward(x)
            if t in T_obs:
                y.append(x)

        return y # returns a list of len(T_obs)!

    def forward(self, x=None, T_obs=None):

        y = [self.model_observer.forward(z) for z in self._forward(x, T_obs)]

        return torch.stack(y, dim=0) # len(T_obs) x N x J+1 x K

    def _sample_obs(self, y, m=None):

        m = [None for t in range(len(y))] if m is None else m
        assert len(m) == len(y)
        self.masks = []
        yn = []
        for i in range(len(y)):
            yn.append(self.model_observer.sample(y[i], m=m[i]))
            self.masks.append(self.model_observer.mask)

        return torch.stack(yn, dim=0) # len(T_obs) x N x J+1 x K

    def sample(self, x=None, m=None, T_obs=None):

        y = self._forward(x, T_obs)
        
        return self._sample_obs(self._forward(x, T_obs), m=m)

    def log_prob(self, y, x=None, m=None, T_obs=None):

        m = self.masks if m is None else m
        xs = self.forward(x, T_obs) # # len(T_obs) x N x J+1 x K

        log_probs = torch.stack([self.model_observer.log_prob(y_, x_, m_) for y_,x_, m_ in zip(y, xs, m)]) # len(T_obs) x N

        return log_probs.sum(axis=0)

    def set_state(self, x_init):

        self.X = torch.nn.Parameter(as_tensor(x_init))

    def set_rollout_len(self, T):

        assert T >= 0
        self.T = T
