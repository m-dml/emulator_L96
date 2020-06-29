import torch
import numpy as np

class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, data, offset=1, 
                 start=None, end=None, 
                 normalize=False, randomize_order=True):

        self.data = data.copy()

        self.offset = offset
        if start is None or end is None:
            start, end = 0,  self.data.shape[0]-self.offset
        assert end > start
        self.start, self.end = start, end

        self.normalize = normalize
        self.mean, self.std = 0., 1.
        if self.normalize:
            self.mean = self.data.mean(axis=0).reshape(1,-1)
            self.std = self.data.std(axis=0).reshape(1,-1)
            self.data = (self.data - self.mean) / self.std 

        self.randomize_order = randomize_order

    def __getitem__(self, index):
        """ Generate one batch of data """
        return np.atleast_2d(self.data[np.asarray(index),:])

    def __iter__(self):
        """ Return iterable over data in random order """
        iter_start, iter_end = self.divide_workers()
        if self.randomize_order:
            idx = torch.randperm(iter_end - iter_start, device='cpu') + iter_start
        else: 
            idx = torch.arange(iter_start, iter_end, requires_grad=False, device='cpu')

        X = self.data[idx,:].reshape(len(idx), 1, -1)
        y = self.data[idx+self.offset,:].reshape(len(idx), 1, -1)

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
