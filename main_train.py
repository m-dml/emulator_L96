import torch
import numpy as np

import os
import sys
module_path = os.path.abspath(os.path.join('/gpfs/home/nonnenma/projects/seasonal_forecasting/code/weatherbench'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.pytorch.layers import setup_conv, ResNetBlock, PeriodicConv2D
from src.pytorch.util import init_torch_device

from L96_emulator.dataset import Dataset
from L96_emulator.networks import TinyNetwork, TinyResNet
from src.pytorch.train import train_model

res_dir = '/gpfs/work/nonnenma/results/emulators/L96/'

device = init_torch_device()
dtype = torch.float32

exp_id = 'V7'

K,J = 36, 10
T, dt = 605, 0.001
temporal_offset = 1
batch_size = 32
T_burnin = int(5./dt) # rough time [s] for model state to have 'forgotten' its initial state

fn_data = f'out_K{K}_J{J}_T{T}'
out = np.load(res_dir + 'data/' + fn_data + '.npy')
print('data.shape', out.shape)

dg_train = Dataset(data=out, J=J, offset=temporal_offset, normalize=True, 
                   start=T_burnin, 
                   end=int(np.floor(out.shape[0]*0.8)))
dg_val   = Dataset(data=out, J=J, offset=temporal_offset, normalize=True, 
                   start=int(np.ceil(out.shape[0]*0.8)), 
                   end=int(np.floor(out.shape[0]*0.9)))

print('N training data:', len(dg_train))
print('N validation data:', len(dg_val))

validation_loader = torch.utils.data.DataLoader(
    dg_val, batch_size=batch_size, drop_last=False, num_workers=0 
)
train_loader = torch.utils.data.DataLoader(
    dg_train, batch_size=batch_size, drop_last=True, num_workers=0
)

save_dir = res_dir + 'models/' + exp_id + '/'
fn_model = f'{exp_id}_FOV5_dt{temporal_offset}.pt'

#model = TinyResNet(n_filters_ks3 = [128, 128], padding_mode='circular')
model = TinyResNet(n_filters_ks3 = [128, 128, 128, 128], 
                    n_channels_in = J+1,
                    n_channels_out = J+1,
                    n_filters_ks1=[[128, 128], [128, 128], [128, 128], [128, 128], [128, 128]],
                    padding_mode='circular')

test_input = np.random.normal(size=(10, J+1, 36))
print(f'model output shape to test input of shape {test_input.shape}', 
      model.forward(torch.as_tensor(test_input, device=device, dtype=dtype)).shape)
print('total #parameters: ', np.sum([np.prod(item.shape) for item in model.state_dict().values()]))

loss_fun = torch.nn.functional.mse_loss

training_outputs = train_model(model, train_loader, validation_loader, device, model_forward=model.forward, loss_fun=loss_fun, 
            lr=0.0001, lr_min=1e-5, lr_decay=0.2, weight_decay=0.,
            max_epochs=200, max_patience=50, max_lr_patience=20, eval_every=2000,
            verbose=True, save_dir=save_dir + fn_model)

training_outputs['training_loss'] = training_outputs['training_loss'][-1]
validation_loss = training_outputs['validation_loss']

np.save(save_dir + '_training_outputs', training_outputs)
      