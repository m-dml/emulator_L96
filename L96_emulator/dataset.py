import torch
import numpy as np
from .util import sortL96intoChannels

class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, data, offset=1, J=0,
                 start=None, end=None, 
                 normalize=False, randomize_order=True):

        if len(data.shape) == 2:
            self.J, self.K = J, data.shape[1]//(J+1)
            assert data.shape[1]/(J+1) == self.K
            self.data = sortL96intoChannels(data, J)

        self.offset = offset
        if start is None or end is None:
            start, end = 0,  self.data.shape[0]-self.offset
        assert end > start
        self.start, self.end = start, end

        self.normalize = normalize
        self.mean, self.std = 0., 1.
        if self.normalize:
            self.mean = self.data.mean(axis=(0,2)).reshape(1,-1,1)
            self.std = self.data.std(axis=(0,2)).reshape(1,-1,1)
            self.data = (self.data - self.mean) / self.std 

        self.randomize_order = randomize_order

    def __getitem__(self, index):
        """ Generate one batch of data """
        idx = np.atleast_1d(np.asarray(index))
        return self.data[idx]

    def __iter__(self):
        """ Return iterable over data in random order """
        iter_start, iter_end = self.divide_workers()
        if self.randomize_order:
            idx = torch.randperm(iter_end - iter_start, device='cpu') + iter_start
        else: 
            idx = torch.arange(iter_start, iter_end, requires_grad=False, device='cpu')

        X = self.data[idx,:]
        y = self.data[idx+self.offset,:]

        return zip(X, y)

    def __len__(self):
        return (self.end - self.start) #self.data.shape[0]

    def divide_workers(self):
        """ parallelized data loading via torch.util.data.Dataloader """
        if torch.utils.data.get_worker_info() is None:
            iter_start = torch.tensor(self.start, requires_grad=False, dtype=torch.int, device='cpu')
            iter_end = torch.tensor(self.end, requires_grad=False, dtype=torch.int, device='cpu')
        else: 
            raise NotImplementedError('had no need for parallelization yet')

        return iter_start, iter_end


class DatasetMultiTrial(Dataset):
    def __init__(self, data, offset=1, J=0,
                 start=None, end=None, 
                 normalize=False, randomize_order=True):

        assert len(data.shape) == 3 # N, T, K*(J+1)
        self.N, self.T = data.shape[:2] # trial count,  trial length
        self.J, self.K = J, data.shape[-1]//(J+1)
        assert data.shape[-1]/(J+1) == self.K
        self.data = sortL96intoChannels(data.reshape(-1,self.K*(self.J+1)),J=J) # N*T, J+1, K


        self.offset = offset
        if start is None or end is None:
            start, end = 0,  self.data.shape[0]-self.offset
        assert end > start
        self.start, self.end = start, end

        self.normalize = normalize
        self.mean, self.std = 0., 1.
        if self.normalize:
            self.mean = self.data.mean(axis=(0,2)).reshape(1,-1,1)
            self.std = self.data.std(axis=(0,2)).reshape(1,-1,1)
            self.data = (self.data - self.mean) / self.std 

        self.randomize_order = randomize_order

    def __getitem__(self, index):
        """ Generate one batch of data """
        idx = np.atleast_1d(np.asarray(index))
        return self.data[idx]

    def __iter__(self):
        """ Return iterable over data in random order """
        iter_start, iter_end = self.divide_workers()
        if self.randomize_order:
            idx = [torch.randperm(iter_end - iter_start, device='cpu') for j in range(self.N)]
            idx = torch.cat([j*self.T + iter_start + idx[j] for j in range(len(idx))])
        else: 
            idx = [torch.arange(iter_start, iter_end, requires_grad=False, device='cpu') for j in range(self.N)]
            idx = torch.cat([j*self.T + idx[j] for j in range(len(idx))])

        X = self.data[idx].reshape(-1,J,K) # reshapes time x n_trials into single axis ! 
        y = self.data[idx+self.offset].reshape(-1,J,K)

        return zip(X, y)

    def __len__(self):
        return (self.end - self.start) #self.data.shape[0]

    def divide_workers(self):
        """ parallelized data loading via torch.util.data.Dataloader """
        if torch.utils.data.get_worker_info() is None:
            iter_start = torch.tensor(self.start, requires_grad=False, dtype=torch.int, device='cpu')
            iter_end = torch.tensor(self.end, requires_grad=False, dtype=torch.int, device='cpu')
        else: 
            raise NotImplementedError('had no need for parallelization yet')

        return iter_start, iter_end


class DatasetRelPred(Dataset):
    def __init__(self, data, offset=1, J=0,
                 start=None, end=None, 
                 normalize=False, randomize_order=True):

        if len(data.shape) == 2:
            self.J, self.K = J, data.shape[1]//(J+1)
            assert data.shape[1]/(J+1) == self.K
            self.data = data.copy().reshape(-1, self.J+1, self.K)

        self.offset = offset
        if start is None or end is None:
            start, end = 0,  self.data.shape[0]-self.offset
        assert end > start
        self.start, self.end = start, end

        self.normalize = normalize
        self.mean, self.std = 0., 1.
        if self.normalize:
            self.mean_in = self.data.mean(axis=(0,2)).reshape(1,-1,1)
            self.std_in = self.data.std(axis=(0,2)).reshape(1,-1,1)
            self.data = (self.data - self.mean_in) / self.std_in

            self.mean_out = np.mean(self.data[:-self.offset] - self.data[self.offset:], axis=(0,2)).reshape(1,-1,1)
            self.std_out =   np.std(self.data[:-self.offset] - self.data[self.offset:], axis=(0,2)).reshape(1,-1,1)

        self.randomize_order = randomize_order

    def __getitem__(self, index):
        """ Generate one batch of data """
        idx = np.atleast_1d(np.asarray(index))
        return self.data[idx] #, (self.data[idx+self.offset,:] - self.data[idx,:] - self.mean_out) / self.std_out

    def __iter__(self):
        """ Return iterable over data in random order """
        iter_start, iter_end = self.divide_workers()
        if self.randomize_order:
            idx = torch.randperm(iter_end - iter_start, device='cpu') + iter_start
        else: 
            idx = torch.arange(iter_start, iter_end, requires_grad=False, device='cpu')

        X = self.data[idx,:]
        y = (self.data[idx+self.offset,:] - self.data[idx,:] - self.mean_out) / self.std_out

        return zip(X, y)

class DatasetRelPredPast(DatasetRelPred):

    def __iter__(self):
        """ Return iterable over data in random order """
        iter_start, iter_end = self.divide_workers()
        if self.randomize_order:
            idx = torch.randperm(iter_end - iter_start, device='cpu') + iter_start
        else: 
            idx = torch.arange(iter_start, iter_end, requires_grad=False, device='cpu')

        X = np.concatenate((self.data[idx,:], self.data[idx,:]-self.data[idx-self.offset,:]), axis=1)
        y = (self.data[idx+self.offset,:] - self.data[idx,:] - self.mean_out) / self.std_out

        return zip(X, y)
