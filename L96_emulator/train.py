import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy


def loss_function(loss_fun, extra_args={}):
    if loss_fun == 'mse':
        return F.mse_loss

    elif loss_fun == 'local_mse':
        n_local = extra_args['n_local']
        pad_local = extra_args['pad_local']
        assert len(pad_local) == 2 # left and right padding along L96 ring of locations

        def local_mse(inputs, targets):
            # inputs.shape  = (N, J+1, K_local+n_local*sum(pad_local))
            # targets.shape = (N, J+1, K_local)
            assert len(inputs.shape)==3
            error = inputs[..., n_local*pad_local[0]:-n_local*pad_local[1]] - targets
            local_mse = torch.sum((error)**2) / inputs.shape[0]            
            return local_mse
        
        return local_mse
    
    elif loss_fun == 'lat_mse':
        # Copied from weatherbench fork of S. Rasp: 
        weights_lat = np.cos(np.deg2rad(extra_args['lat']))
        weights_lat /= weights_lat.mean()
        weights_lat = torch.tensor(weights_lat, requires_grad=False)

        def weighted_mse(in1, in2):
            error = in1 - in2
            weighted_mse = (error)**2 * weights_lat[None, None , :, None]
            weighted_mse = torch.sum(weighted_mse) / in1.shape[0]            
            return weighted_mse

        return weighted_mse

    else:
        raise NotImplementedError()


def calc_val_loss(validation_loader, model_forward, device, loss_fun=F.mse_loss):
    val_loss = 0.0
    with torch.no_grad():
        nb = 0
        for batch in validation_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            val_loss += loss_fun(model_forward(inputs), targets).item()
            nb += 1
    return val_loss / nb


def train_model(model, train_loader, validation_loader, device, model_forward, loss_fun=F.mse_loss, 
                lr=0.001, lr_min=1e-5, lr_decay=0.2, weight_decay=0.,
                max_epochs=200, max_patience=20, max_lr_patience=5, eval_every=None,
                verbose=True, save_dir=None, model_fn=None, output_fn=None):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    def lr_getter():
        return optimizer.param_groups[0]['lr']
    def lr_setter(lr):
        optimizer.param_groups[0]['lr'] = lr

    n_batches = len(train_loader) // train_loader.batch_size
    if not train_loader.drop_last and len(train_loader) > n_batches * train_loader.batch_size:
        n_batches += 1
    eval_every = n_batches if eval_every is None else eval_every

    best_loss, patience, lr_patience = np.inf, max_patience, max_lr_patience
    best_state_dict = {}

    training_loss = []  # list (over epochs) of numpy arrays (over minibatches)
    validation_loss = np.zeros(0, dtype=np.float32)

    epoch, num_steps = 0, 0
    model.train()
    while True:

        epoch += 1
        if epoch > max_epochs:
            break

        training_loss.append(np.full(n_batches, np.nan, dtype=np.float32))

        # Train for a single epoch.
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = batch[0].to(device), batch[1].to(device)
            loss = loss_fun(model_forward(inputs), targets)
            loss.backward()
            optimizer.step()
            num_steps += 1

            training_loss[-1][batch_index] = loss.item()  # record training loss
            assert np.isfinite(training_loss[-1][batch_index])

            if np.mod(num_steps, eval_every) == 0:
                # Track convergence on validation set
                validation_loss = np.append(validation_loss, 
                                            calc_val_loss(validation_loader, model_forward, device, loss_fun)
                                           )
                if verbose:
                    print(f'epoch #{epoch} || loss (last batch) {loss} || validation loss {validation_loss[-1]}')

                if validation_loss[-1] < best_loss:
                    patience, lr_patience = max_patience, max_lr_patience
                    best_loss = validation_loss[-1]
                    best_state_dict = deepcopy(model.state_dict())
                    if not save_dir is None and not model_fn is None:
                        torch.save(best_state_dict, save_dir + model_fn)
                else:
                    patience -= 1
                    lr_patience -= 1

                if lr_patience <= 0 and lr_getter() >= lr_min:
                    lr_setter(lr_getter() * lr_decay)
                    print('setting new lr :', str(lr_getter()))
                    lr_patience = max_lr_patience

        if not save_dir is None and not output_fn is None:
            print('saving training outputs to ' + save_dir +  output_fn + '.npy')
            np.save(save_dir + output_fn, dict(training_loss=training_loss, validation_loss=validation_loss))

        if patience <= 0:
            break

    model.load_state_dict(best_state_dict)

    return dict(training_loss=training_loss, validation_loss=validation_loss)