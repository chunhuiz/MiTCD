import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from models.model_helper import activation_helper
from models.CAGKE import CAGKE_1, CAGKE_fix, CAGKE_learnable,CAGKE_learnable_minmax
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score


class MLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation):
        super(MLP, self).__init__()
        self.activation = activation_helper(activation)

        # Set up network.
        layer = nn.Conv1d(num_series, hidden[0], lag)
        modules = [layer]

        for d_in, d_out in zip(hidden, hidden[1:] + [1]):
            layer = nn.Conv1d(d_in, d_out, 1)
            modules.append(layer)

        # Register parameters.
        self.layers = nn.ModuleList(modules)

    def forward(self, X):
        X = X.transpose(2, 1)
        for i, fc in enumerate(self.layers):
            if i != 0:
                X = self.activation(X)
            X = fc(X)

        return X.transpose(2, 1)




class CAGKE_U(nn.Module):
    '''
    CAGKE for discrete variable
    input : X of shape (batch,length_per_batch,p)

    '''

    def __init__(self, length, embed_dim, sigma_min=0.4, sigma_max=4, noise_sigma=0.01, norm_flag=True):
        super(CAGKE_U, self).__init__()

        self.CAGKE_unit = CAGKE_learnable_minmax(in_length=length, embed_dim=embed_dim, sigma_min=sigma_min,
                                          sigma_max=sigma_max, noise_sigma=noise_sigma)

        self.norm_flag = norm_flag

    def forward(self, X, type_flag):
        X_con = torch.zeros_like(X)
        for i in range(X.shape[-1]):
            if type_flag[i]:
                for j in range(X.shape[0]):
                    X_con[j, :, i] = X[j, :, i]
            if not type_flag[i]:
                for j in range(X.shape[0]):
                    X_con[j, :, i] = self.CAGKE_unit(X[j, :, i].unsqueeze(0)).squeeze(0)
        return X_con


class CAGKE_Multi(nn.Module):
    '''
    CAGKE for discrete variable
    input : X of shape (batch,length_per_batch,p)

    '''

    def __init__(self, length, num_dis, embed_dim, sigma_min=0.4, sigma_max=4, noise_sigma=0.01, norm_flag=True):
        super(CAGKE_Multi, self).__init__()

        self.CAGKE_units = nn.ModuleList([CAGKE_learnable_minmax(in_length=length, embed_dim=embed_dim, sigma_min=sigma_min,
                                                          sigma_max=sigma_max, noise_sigma=noise_sigma) for _ in
                                          range(num_dis)])

        self.norm_flag = norm_flag

    def forward(self, X, type_flag):
        X_con = torch.zeros_like(X)
        dis_ind = 0
        for i in range(X.shape[-1]):
            if type_flag[i]:
                for j in range(X.shape[0]):
                    X_con[j, :, i] = X[j, :, i]
            if not type_flag[i]:
                for j in range(X.shape[0]):
                    X_con[j, :, i] = self.CAGKE_units[dis_ind](X[j, :, i].unsqueeze(0)).squeeze(0)
                dis_ind = dis_ind + 1

        return X_con



class cMLP(nn.Module):
    def __init__(self, num_series, lag, hidden, embed_dim, length, activation='relu'):
        '''
        cMLP model with one MLP per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLP, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)

        # Set up networks.
        # self.CAGKE = CAGKE_U(length = length, embed_dim = embed_dim)
        self.networks = nn.ModuleList([
            MLP(num_series, lag, hidden, activation)
            for _ in range(num_series)])

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
          type_flag: indicate weather the variable is continous or discrete of shape (p) 1(continous) 0(discrete)
        '''
        # X_con=self.CAGKE(X,type_flag)

        return torch.cat([network(X) for network in self.networks], dim=2)

    def GC(self, threshold=True, ignore_lag=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.
          ignore_lag: if true, calculate norm of weights jointly for all lags.

        Returns:
          GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
            indicates whether variable j is Granger causal of variable i. In
            second case, entry (i, j, k) indicates whether it's Granger causal
            at lag k.
        '''
        if ignore_lag:
            GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
                  for net in self.networks]
        else:
            GC = [torch.norm(net.layers[0].weight, dim=0)
                  for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC


class Decoder_rec_uni(nn.Module):
    '''
    MLP for dis-variable self-reconstruction
    input : specific uniariate series of shape [batch, length_per_batch,1]
    output: reconstructed series of shaoe [batch,length_per_batch,1,2]
    '''

    def __init__(self, lag, hidden, activation):
        super(Decoder_rec_uni, self).__init__()
        self.activation = activation_helper(activation)

        # Set up network.
        layer = nn.Conv1d(1, hidden[0], lag)
        modules = [layer]

        # only for reconstruction
        # two-class classification
        for d_in, d_out in zip(hidden, hidden[1:] + [2]):
            layer = nn.Conv1d(d_in, d_out, 1)
            modules.append(layer)

        # Register parameters.
        self.layers = nn.ModuleList(modules)

    def forward(self, X):
        X = X.transpose(2, 1)
        for i, fc in enumerate(self.layers):
            if i != 0:
                X = self.activation(X)
            X = fc(X)

        return X.transpose(2, 1).unsqueeze(-2)


class DREC(nn.Module):
    def __init__(self, num_dis_series, lag, hidden, embed_dim, length, activation='relu'):
        '''

        Discrete variable decoder (one corresponds to one)


        '''
        super(DREC, self).__init__()

        self.lag = lag
        self.activation = activation_helper(activation)

        # Set up networks.
        # self.CAGKE = CAGKE_U(length = length, embed_dim = embed_dim)
        self.networks = nn.ModuleList([
            Decoder_rec_uni(lag, hidden, activation)
            for _ in range(num_dis_series)])

    def forward(self, X, type_flag):
        '''
        Perform forward pass for reconstruction.

        Args:
          X: torch tensor of shape (batch, T, p).
          X_dis: shape (batch, T, p-n)
          type_flag: indicate weather the variable is continous or discrete of shape (p) 1(continous) 0(discrete)

        return:

         reconstructed TS of shaoe (batch,T-lag+1,dis_dim,2)
        '''

        # X_con=self.CAGKE(X,type_flag)
        X_recons = torch.zeros([X.shape[0], X.shape[1] - self.lag + 1, len(type_flag) - np.sum(type_flag), 2]).to(
            X.device)

        ind = 0
        for i in range(len(type_flag)):
            if type_flag[i] == 0:
                X_recons[:, :, [ind], :] = self.networks[ind](X[:, :, [i]])
                ind = ind + 1

        return X_recons.to(X.device)


def prox_update(network, lam, lr, penalty):
    '''
    Perform in place proximal update on first layer weight matrix.

    Args:
      network: MLP network.
      lam: regularization parameter.
      lr: learning rate.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL':
        norm = torch.norm(W, dim=0, keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        for i in range(lag):
            norm = torch.norm(W[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
            W.data[:, :, :(i + 1)] = (
                    (W.data[:, :, :(i + 1)] / torch.clamp(norm, min=(lr * lam)))
                    * torch.clamp(norm - (lr * lam), min=0.0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def regularize(network, lam, penalty):
    '''
    Calculate regularization term for first layer weight matrix.

    Args:
      network: MLP network.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(W, dim=(0, 2)))
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(W, dim=(0, 2)))
                      + torch.sum(torch.norm(W, dim=0)))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        return lam * sum([torch.sum(torch.norm(W[:, :, :(i + 1)], dim=(0, 2)))
                          for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def ridge_regularize(network, lam):
    '''Apply ridge penalty at all subsequent layers.'''
    return lam * sum([torch.sum(fc.weight ** 2) for fc in network.layers[1:]])


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def pre_train_model_v2(cagke, cmlp, drec, X, type_flag, lr, max_iter, multi_cagke=False, rec_lambda=0.1,
                       lam=0, lam_ridge=0, penalty='H',
                       lookback=20, check_every=100, verbose=1):
    '''Pre-Train model with Adam'''
    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    loss_fn_d = nn.CrossEntropyLoss()

    # ！！！
    optimizer_nets = torch.optim.Adam(cmlp.parameters(), lr=lr)
    optimizer_cagke = torch.optim.Adam(cagke.parameters(), lr=lr)
    optimizer_drec = torch.optim.Adam(drec.parameters(), lr=lr * 5)
    train_loss_list = []
    forecast_loss_list = []
    recons_loss_list = []
    # div_loss_list = []
    recon_acc_list = []
    cagke_sigma_list = []
    cagke_weight_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = deepcopy(cmlp)

    print('----Start pre training :) ----')
    print('----Continuous variables are:----')

    best_cagke = deepcopy(cagke)
    best_drec = deepcopy(cagke)

    print(np.where(np.array(type_flag) == 1)[0].tolist())

    '''
    prepare dis-variable data for self-supervised reconstruction [batsh_size,T-lag+1,p_dis]
    '''
    dis_recon_pre = torch.zeros([X.shape[0], X.shape[1] - lag + 1, len(type_flag) - np.sum(type_flag)])

    ind = 0
    for i in range(len(type_flag)):
        if type_flag[i] == 0:
            dis_recon_pre[:, :, ind] = X[:, lag - 1:, i]
            ind = ind + 1

    dis_recon_pre = dis_recon_pre.to(X.device)

    for it in range(max_iter):

        if multi_cagke:
            cagke_sigma_list.append([cagke.CAGKE_units[0].learnable_sigma_min.data.detach().cpu().numpy(),
                                    cagke.CAGKE_units[0].learnable_sigma_max.data.detach().cpu().numpy()])

            cagke_weight_list.append(cagke.CAGKE_units[0].learnable_weight.data.detach().cpu().numpy())

        else:
            cagke_sigma_list.append([cagke.CAGKE_unit.learnable_sigma_min.data.detach().cpu().numpy(),
                                    cagke.CAGKE_unit.learnable_sigma_max.data.detach().cpu().numpy()])

            cagke_weight_list.append(cagke.CAGKE_unit.learnable_weight.data.detach().cpu().numpy())

        X_con = cagke(X, type_flag)
        # X_con_dis = X_con

        forecast_loss = sum([loss_fn(cmlp.networks[i](X_con[:, :-1]), X[:, lag:, i:i + 1])
                             for i in np.where(np.array(type_flag) == 1)[0].tolist()])

        recon_loss = loss_fn_d(drec(X_con, type_flag).reshape(-1, 2), dis_recon_pre.reshape(-1).long())


        loss = forecast_loss + rec_lambda * recon_loss

        # Add penalty terms.
        if lam > 0:
            loss = loss + sum([regularize(net, lam, penalty)
                               for net in cmlp.networks])
        if lam_ridge > 0:
            loss = loss + sum([ridge_regularize(net, lam_ridge)
                               for net in cmlp.networks])

        # Take gradient step.

        loss.backward()
        optimizer_nets.step()
        optimizer_cagke.step()
        optimizer_drec.step()

        cmlp.zero_grad()
        cagke.zero_grad()
        drec.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            mean_loss = loss / p
            train_loss_list.append(mean_loss.detach())
            forecast_loss_list.append(forecast_loss.detach())
            recons_loss_list.append(recon_loss.detach())
            # div_loss_list.append(band_div_loss.detach())
            recon_acc_list.append(
                dis_recon_pre.clone().eq(drec(X_con, type_flag).clone().argmax(dim=-1)).float().mean().detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)
                print('forecasting_Loss = %f' % forecast_loss)
                print('recons_Loss = %f' % recon_loss)
                print('reconstruction_acc = %f' % dis_recon_pre.clone().eq(
                    drec(X_con, type_flag).clone().argmax(dim=-1)).float().mean().detach().cpu().numpy())
                # print('diversity_Loss = %f' % band_div_loss)

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it

                best_model = deepcopy(cmlp)
                best_cagke = deepcopy(cagke)
                best_drec = deepcopy(drec)

            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(cmlp, best_model)

    restore_parameters(cagke, best_cagke)
    restore_parameters(drec, best_drec)

    return train_loss_list, forecast_loss_list, recons_loss_list, recon_acc_list, cagke_sigma_list, cagke_weight_list


def train_model_formal(cagke, cmlp, drec, X, type_flag, lam, lam_ridge, lr, penalty,
                       max_iter, GC_True, multi_cagke=False, cagke_learn_flag=True, cagke_up_period=100, rec_lambda=0.1,
                       check_every=100, r=0.8, lr_min=1e-8, sigma=0.5,
                       monotone=False, m=10, lr_decay=0.5,
                       begin_line_search=True, switch_tol=1e-3, verbose=1):
    '''
    Train cMLP model with GISTA.

    Args:
      cagke : pseudo-continuous transformation
      cmlp: cmlp model.
      drec: discrete variable decoder
      X: tensor of data, shape (batch, T, p).
      lam: parameter for nonsmooth regularization.
      lam_ridge: parameter for ridge regularization on output layer.
      lr: learning rate.
      penalty: type of nonsmooth regularization.
      max_iter: max number of GISTA iterations.
      check_every: how frequently to record loss.
      r: for line search.
      lr_min: for line search.
      sigma: for line search.
      monotone: for line search.
      m: for line search.
      lr_decay: for adjusting initial learning rate of line search.
      begin_line_search: whether to begin with line search.
      switch_tol: tolerance for switching to line search.
      verbose: level of verbosity (0, 1, 2).
    '''

    p = cmlp.p
    lag = cmlp.lag
    cmlp_copy = deepcopy(cmlp)

    loss_fn = nn.MSELoss(reduction='mean')
    lr_list = [lr for _ in range(p)]

    '''
       Prepare dis-variable data for self-supervised reconstruction [batsh_size,T-lag+1,p_dis]
    '''
    dis_recon_pre = torch.zeros([X.shape[0], X.shape[1] - lag + 1, len(type_flag) - np.sum(type_flag)])

    ind = 0
    for i in range(len(type_flag)):
        if type_flag[i] == 0:
            dis_recon_pre[:, :, ind] = X[:, lag - 1:, ind]
            ind = ind + 1

    dis_recon_pre = dis_recon_pre.to(X.device).detach()

    print('----Learning rate list:----')
    print(lr_list)

    # Calculate full loss.
    mse_list = []
    smooth_list = []
    loss_list = []

    X_con = cagke(X, type_flag).detach()

    # calculate reconstruction loss
    loss_fn_d = nn.CrossEntropyLoss()

    print('----Start formal training :) ----')
    auroc_best = 0
    auprc_best = 0
    best_iter = 0

    best_model = deepcopy(cmlp)
    best_cagke = deepcopy(cagke)
    best_drec = deepcopy(drec)

    for i in range(p):
        net = cmlp.networks[i]
        mse = loss_fn(net(X_con[:, :-1]), X_con[:, lag:, i: i + 1].detach())
        ridge = ridge_regularize(net, lam_ridge)
        smooth = mse + ridge

        '''
        Add discrete variable reconstruction loss

        '''

        mse_list.append(mse)
        smooth_list.append(smooth)
        with torch.no_grad():
            nonsmooth = regularize(net, lam, penalty)
            loss = smooth + nonsmooth
            loss_list.append(loss)

    # Set up lists for loss and mse.
    with torch.no_grad():
        loss_mean = sum(loss_list) / p
        mse_mean = sum(mse_list) / p

    train_loss_list = [loss_mean]
    train_mse_list = [mse_mean]

    # For switching to line search.
    line_search = begin_line_search

    # For line search criterion.
    done = [False for _ in range(p)]
    assert 0 < sigma <= 1
    assert m > 0
    if not monotone:
        last_losses = [[loss_list[i]] for i in range(p)]

    for it in range(max_iter):
        # Backpropagate errors.
        sum([smooth_list[i] for i in range(p) if not done[i]]).backward()

        # For next iteration.
        new_mse_list = []
        new_smooth_list = []
        new_loss_list = []

        # Perform GISTA step for each network.

        dis_idxx = 0
        for i in range(p):
            # Skip if network converged.
            if done[i]:
                new_mse_list.append(mse_list[i])
                new_smooth_list.append(smooth_list[i])
                new_loss_list.append(loss_list[i])
                continue

            # Prepare for line search.
            step = False
            lr_it = lr_list[i]
            net = cmlp.networks[i]
            net_copy = cmlp_copy.networks[i]
            cagke_net = cagke

            while not step:
                # Perform tentative ISTA step.
                for param, temp_param in zip(net.parameters(), net_copy.parameters()):
                    temp_param.data = param - lr_it * param.grad

                # learnable weight
                if it % cagke_up_period == 0:
                    if cagke_learn_flag:
                        if multi_cagke:
                            for unit in cagke.CAGKE_units:
                                unit.learnable_weight.data = unit.learnable_weight \
                                                             - 0.1 * lr_it * unit.learnable_weight.grad


                        else:
                            cagke_net.CAGKE_unit.learnable_weight.data = cagke_net.CAGKE_unit.learnable_weight \
                                                                         - 0.1 * lr_it * cagke_net.CAGKE_unit.learnable_weight.grad

                # Proximal update.
                prox_update(net_copy, lam, lr_it, penalty)

                # Check line search criterion

                # update psedudo-continity transformation with period-wise optimization
                if (it + 1) % cagke_up_period == 0:
                    X_con = cagke(X, type_flag)
                else:
                    X_con = X_con.detach()

                mse = loss_fn(net_copy(X_con[:, :-1]), X_con[:, lag:, i:i + 1].detach())

                ridge = ridge_regularize(net_copy, lam_ridge)

                smooth = mse + ridge

                # add reconstruction loss
                if type_flag[i] == 0:
                    # print('ok')
                    # print(i)
                    rec_loss = loss_fn_d(drec(X_con, type_flag)[:, :, dis_idxx, :].reshape(-1, 2),
                                         dis_recon_pre[:, :, dis_idxx].reshape(-1).long())

                    if (it + 1) % cagke_up_period == 0:
                        smooth = smooth + rec_lambda * rec_loss.detach()
                    else:
                        smooth = smooth + rec_lambda * rec_loss

                with torch.no_grad():
                    nonsmooth = regularize(net_copy, lam, penalty)
                    loss = smooth + nonsmooth

                    # Elimate the influence of reconstruction loss for converged decision
                    if type_flag[i] == 0:
                        loss = loss - rec_lambda * rec_loss.detach()

                    tol = (0.5 * sigma / lr_it) * sum(
                        [torch.sum((param - temp_param) ** 2)
                         for param, temp_param in
                         zip(net.parameters(), net_copy.parameters())])

                comp = loss_list[i] if monotone else max(last_losses[i])
                if not line_search or (comp - loss) > tol:
                    step = True
                    if verbose > 1:
                        print('Taking step, network i = %d, lr = %f'
                              % (i, lr_it))
                        print('Gap = %f, tol = %f' % (comp - loss, tol))

                    # For next iteration.
                    new_mse_list.append(mse)
                    new_smooth_list.append(smooth)
                    new_loss_list.append(loss)

                    # Adjust initial learning rate.
                    lr_list[i] = (
                            (lr_list[i] ** (1 - lr_decay)) * (lr_it ** lr_decay))

                    if not monotone:
                        if len(last_losses[i]) == m:
                            last_losses[i].pop(0)
                        last_losses[i].append(loss)

                else:
                    # Reduce learning rate.
                    lr_it *= r
                    if lr_it < lr_min:
                        done[i] = True
                        new_mse_list.append(mse_list[i])
                        new_smooth_list.append(smooth_list[i])
                        new_loss_list.append(loss_list[i])
                        if verbose > 0:
                            print('Network %d converged' % (i + 1))
                        break

            # Clean up.
            net.zero_grad()

            if step:
                # Swap network parameters.
                cmlp.networks[i], cmlp_copy.networks[i] = net_copy, net
                cagke = cagke_net

            # update dis_re_index
            if type_flag[i] == 0:
                dis_idxx = dis_idxx + 1

        # For next iteration.
        mse_list = new_mse_list
        smooth_list = new_smooth_list
        loss_list = new_loss_list

        # Check if all networks have converged.
        if sum(done) == p:
            if verbose > 0:
                print('Done at iteration = %d' % (it + 1))
            break

        # Check progress.
        if (it + 1) % check_every == 0:
            with torch.no_grad():
                loss_mean = sum(loss_list) / p
                mse_mean = sum(mse_list) / p
                ridge_mean = (sum(smooth_list) - sum(mse_list)) / p
                nonsmooth_mean = (sum(loss_list) - sum(smooth_list)) / p

                GC_estimate = cmlp.GC(threshold=False).cpu().data.numpy()
                AUROC = roc_auc_score(GC_True.reshape([-1, 1]), GC_estimate.reshape([-1, 1]))
                AUPRC = average_precision_score(GC_True.reshape([-1, 1]), GC_estimate.reshape([-1, 1]))

                if AUROC > auroc_best:
                    auroc_best = AUROC

                    best_model = deepcopy(cmlp)
                    best_cagke = deepcopy(cagke)
                    best_drec = deepcopy(drec)

                if AUPRC > auprc_best:
                    auprc_best = AUPRC

            train_loss_list.append(loss_mean)
            train_mse_list.append(mse_mean)

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Total loss = %f' % loss_mean)
                print('MSE = %f, Ridge = %f, Nonsmooth = %f'
                      % (mse_mean, ridge_mean, nonsmooth_mean))
                print('Variable usage = %.2f%%'
                      % (100 * torch.mean(cmlp.GC().float())))

                print('AUROC= %.4f%%' % (100 * AUROC))
                print('AUPRC= %.4f%%' % (100 * AUPRC))

            # Check whether loss has increased.
            if not line_search:
                if train_loss_list[-2] - train_loss_list[-1] < switch_tol:
                    line_search = True
                    if verbose > 0:
                        print('Switching to line search')


    restore_parameters(cmlp, best_model)
    restore_parameters(cagke, best_cagke)
    restore_parameters(drec, best_drec)

    return train_loss_list, train_mse_list, auroc_best, auprc_best


def train_model_formal_for_vis(cagke, cmlp, drec, X, type_flag, lam, lam_ridge, lr, penalty,
                       max_iter, GC_True, multi_cagke=False, cagke_learn_flag=True, cagke_up_period=100, rec_lambda=0.1,
                       check_every=100, r=0.8, lr_min=1e-8, sigma=0.5,
                       monotone=False, m=10, lr_decay=0.5,
                       begin_line_search=True, switch_tol=1e-3, verbose=1):
    '''
    Train cMLP model with GISTA.

    Args:
      cagke : pseudo-continuous transformation
      cmlp: cmlp model.
      drec: discrete variable decoder
      X: tensor of data, shape (batch, T, p).
      lam: parameter for nonsmooth regularization.
      lam_ridge: parameter for ridge regularization on output layer.
      lr: learning rate.
      penalty: type of nonsmooth regularization.
      max_iter: max number of GISTA iterations.
      check_every: how frequently to record loss.
      r: for line search.
      lr_min: for line search.
      sigma: for line search.
      monotone: for line search.
      m: for line search.
      lr_decay: for adjusting initial learning rate of line search.
      begin_line_search: whether to begin with line search.
      switch_tol: tolerance for switching to line search.
      verbose: level of verbosity (0, 1, 2).
    '''

    p = cmlp.p
    lag = cmlp.lag
    cmlp_copy = deepcopy(cmlp)

    loss_fn = nn.MSELoss(reduction='mean')
    lr_list = [lr for _ in range(p)]

    '''
       Prepare dis-variable data for self-supervised reconstruction [batsh_size,T-lag+1,p_dis]
    '''
    dis_recon_pre = torch.zeros([X.shape[0], X.shape[1] - lag + 1, len(type_flag) - np.sum(type_flag)])

    ind = 0
    for i in range(len(type_flag)):
        if type_flag[i] == 0:
            dis_recon_pre[:, :, ind] = X[:, lag - 1:, ind]
            ind = ind + 1

    dis_recon_pre = dis_recon_pre.to(X.device).detach()

    print('----Learning rate list:----')
    print(lr_list)

    # Calculate full loss.
    mse_list = []
    # recons_list = []
    smooth_list = []
    loss_list = []

    X_con = cagke(X, type_flag).detach()

    # calculate reconstruction loss
    loss_fn_d = nn.CrossEntropyLoss()

    print('----Start formal training :) ----')
    auroc_best = 0
    auprc_best = 0
    auroc_list = []
    auprc_list = []
    non_smooth_list = []
    best_iter = 0

    best_model = deepcopy(cmlp)
    best_cagke = deepcopy(cagke)
    best_drec = deepcopy(drec)

    for i in range(p):
        net = cmlp.networks[i]
        mse = loss_fn(net(X_con[:, :-1]), X_con[:, lag:, i: i + 1].detach())
        ridge = ridge_regularize(net, lam_ridge)
        smooth = mse + ridge

        '''
        Add discrete variable reconstruction loss

        '''

        # dis_idx=0
        # if type_flag[i] == 0:
        #
        #     rec_loss = loss_fn_d(drec(X_con, type_flag)[:,:,dis_idx,:].reshape(-1, 2), dis_recon_pre[:,:,dis_idx].reshape(-1).long())
        #     smooth = smooth + rec_lambda * rec_loss
        #     dis_idx = dis_idx + 1

        mse_list.append(mse)
        smooth_list.append(smooth)
        with torch.no_grad():
            nonsmooth = regularize(net, lam, penalty)
            loss = smooth + nonsmooth
            loss_list.append(loss)

    # Set up lists for loss and mse.
    with torch.no_grad():
        loss_mean = sum(loss_list) / p
        mse_mean = sum(mse_list) / p

    train_loss_list = [loss_mean]
    train_mse_list = [mse_mean]

    # For switching to line search.
    line_search = begin_line_search

    # For line search criterion.
    done = [False for _ in range(p)]
    assert 0 < sigma <= 1
    assert m > 0
    if not monotone:
        last_losses = [[loss_list[i]] for i in range(p)]

    for it in range(max_iter):
        # Backpropagate errors.
        sum([smooth_list[i] for i in range(p) if not done[i]]).backward()

        # For next iteration.
        new_mse_list = []
        new_smooth_list = []
        new_loss_list = []

        # Perform GISTA step for each network.

        dis_idxx = 0
        for i in range(p):
            # Skip if network converged.
            if done[i]:
                new_mse_list.append(mse_list[i])
                new_smooth_list.append(smooth_list[i])
                new_loss_list.append(loss_list[i])
                continue

            # Prepare for line search.
            step = False
            lr_it = lr_list[i]
            net = cmlp.networks[i]
            net_copy = cmlp_copy.networks[i]
            cagke_net = cagke

            while not step:
                # Perform tentative ISTA step.
                for param, temp_param in zip(net.parameters(), net_copy.parameters()):
                    temp_param.data = param - lr_it * param.grad

                # learnable weight
                if it % cagke_up_period == 0:
                    if cagke_learn_flag:
                        if multi_cagke:
                            for unit in cagke.CAGKE_units:
                                unit.learnable_weight.data = unit.learnable_weight \
                                                             - 0.1 * lr_it * unit.learnable_weight.grad


                        else:
                            cagke_net.CAGKE_unit.learnable_weight.data = cagke_net.CAGKE_unit.learnable_weight \
                                                                         - 0.1 * lr_it * cagke_net.CAGKE_unit.learnable_weight.grad

                # Proximal update.
                prox_update(net_copy, lam, lr_it, penalty)

                # Check line search criterion

                # update psedudo-continity transformation with period-wise optimization
                if (it + 1) % cagke_up_period == 0:
                    X_con = cagke(X, type_flag)
                else:
                    X_con = X_con.detach()

                mse = loss_fn(net_copy(X_con[:, :-1]), X_con[:, lag:, i:i + 1].detach())

                ridge = ridge_regularize(net_copy, lam_ridge)

                smooth = mse + ridge

                # add reconstruction loss
                if type_flag[i] == 0:
                    # print('ok')
                    # print(i)
                    rec_loss = loss_fn_d(drec(X_con, type_flag)[:, :, dis_idxx, :].reshape(-1, 2),
                                         dis_recon_pre[:, :, dis_idxx].reshape(-1).long())

                    if (it + 1) % cagke_up_period == 0:
                        smooth = smooth + rec_lambda * rec_loss.detach()
                    else:
                        smooth = smooth + rec_lambda * rec_loss

                with torch.no_grad():
                    nonsmooth = regularize(net_copy, lam, penalty)
                    loss = smooth + nonsmooth

                    # Elimate the influence of reconstruction loss for converged decision
                    if type_flag[i] == 0:
                        loss = loss - rec_lambda * rec_loss.detach()

                    tol = (0.5 * sigma / lr_it) * sum(
                        [torch.sum((param - temp_param) ** 2)
                         for param, temp_param in
                         zip(net.parameters(), net_copy.parameters())])

                comp = loss_list[i] if monotone else max(last_losses[i])
                if not line_search or (comp - loss) > tol:
                    step = True
                    if verbose > 1:
                        print('Taking step, network i = %d, lr = %f'
                              % (i, lr_it))
                        print('Gap = %f, tol = %f' % (comp - loss, tol))

                    # For next iteration.
                    new_mse_list.append(mse)
                    new_smooth_list.append(smooth)
                    new_loss_list.append(loss)

                    # Adjust initial learning rate.
                    lr_list[i] = (
                            (lr_list[i] ** (1 - lr_decay)) * (lr_it ** lr_decay))

                    if not monotone:
                        if len(last_losses[i]) == m:
                            last_losses[i].pop(0)
                        last_losses[i].append(loss)

                else:
                    # Reduce learning rate.
                    lr_it *= r
                    if lr_it < lr_min:
                        done[i] = True
                        new_mse_list.append(mse_list[i])
                        new_smooth_list.append(smooth_list[i])
                        new_loss_list.append(loss_list[i])
                        if verbose > 0:
                            print('Network %d converged' % (i + 1))
                        break

            # Clean up.
            net.zero_grad()

            if step:
                # Swap network parameters.
                cmlp.networks[i], cmlp_copy.networks[i] = net_copy, net
                cagke = cagke_net

            # update dis_re_index
            if type_flag[i] == 0:
                dis_idxx = dis_idxx + 1

        # For next iteration.
        mse_list = new_mse_list
        smooth_list = new_smooth_list
        loss_list = new_loss_list

        # Check if all networks have converged.
        if sum(done) == p:
            if verbose > 0:
                print('Done at iteration = %d' % (it + 1))
            break

        # Check progress.
        if (it + 1) % check_every == 0:
            with torch.no_grad():
                loss_mean = sum(loss_list) / p
                mse_mean = sum(mse_list) / p
                ridge_mean = (sum(smooth_list) - sum(mse_list)) / p
                nonsmooth_mean = (sum(loss_list) - sum(smooth_list)) / p

                GC_estimate = cmlp.GC(threshold=False).cpu().data.numpy()
                AUROC = roc_auc_score(GC_True.reshape([-1, 1]), GC_estimate.reshape([-1, 1]))
                AUPRC = average_precision_score(GC_True.reshape([-1, 1]), GC_estimate.reshape([-1, 1]))

                if AUROC > auroc_best:
                    auroc_best = AUROC

                    best_model = deepcopy(cmlp)
                    best_cagke = deepcopy(cagke)
                    best_drec = deepcopy(drec)

                if AUPRC > auprc_best:
                    auprc_best = AUPRC

            train_loss_list.append(loss_mean)
            train_mse_list.append(mse_mean)
            auroc_list.append(AUROC)
            auprc_list.append(AUPRC)
            non_smooth_list.append(nonsmooth_mean)

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Total loss = %f' % loss_mean)
                print('MSE = %f, Ridge = %f, Nonsmooth = %f'
                      % (mse_mean, ridge_mean, nonsmooth_mean))
                print('Variable usage = %.2f%%'
                      % (100 * torch.mean(cmlp.GC().float())))

                print('AUROC= %.4f%%' % (100 * AUROC))
                print('AUPRC= %.4f%%' % (100 * AUPRC))

            # Check whether loss has increased.
            if not line_search:
                if train_loss_list[-2] - train_loss_list[-1] < switch_tol:
                    line_search = True
                    if verbose > 0:
                        print('Switching to line search')


    restore_parameters(cmlp, best_model)
    restore_parameters(cagke, best_cagke)
    restore_parameters(drec, best_drec)

    return train_loss_list, train_mse_list, auroc_best, auprc_best,auroc_list,auprc_list,non_smooth_list

