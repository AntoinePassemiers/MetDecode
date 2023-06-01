# -*- coding: utf-8 -*-
# metdecode2.py
# author: Antoine Passemiers

from typing import Tuple, Any

import numpy as np
import scipy.optimize
import scipy.special
import torch
import torch.nn.functional


# torch.autograd.set_detect_anomaly(True)


class SideInformationModule(torch.nn.Module):

    def __init__(self, n_tissues: int, n_confounders: int, n_markers: int):
        torch.nn.Module.__init__(self)
        self.n_tissues: int = n_tissues
        self.n_confounders: int = n_confounders
        self.n_inputs: int = self.n_tissues + self.n_confounders
        self.n_outputs: int = self.n_tissues

        self.tissue_bias = torch.nn.Parameter(torch.FloatTensor(np.asarray([0.001] * n_tissues)))
        self.marker_bias = torch.nn.Parameter(torch.FloatTensor(np.asarray([0.001] * n_markers)))
        self.layers: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(self.n_inputs, self.n_inputs),
            torch.nn.Tanh(),
            torch.nn.Linear(self.n_inputs, self.n_inputs),
            torch.nn.Tanh(),
            torch.nn.Linear(self.n_inputs, self.n_outputs),
        )
        n_latent = 1
        self.adapter: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(self.n_tissues, n_latent),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(n_latent, self.n_tissues),
        )
        self.apply(SideInformationModule.init_weights)

    def adapt(self, gamma_logit: torch.Tensor) -> torch.Tensor:
        #print(gamma_logit.size(), self.marker_bias.size(), self.tissue_bias.size(), self.n_tissues, self.n_confounders, self.n_inputs, self.n_outputs)
        #adapted = self.adapter(gamma.t()).t()
        bias = self.marker_bias.unsqueeze(0) + self.tissue_bias.unsqueeze(1)
        adapted = gamma_logit + bias
        return adapted

    def forward(self, gamma_logit: torch.Tensor, confounders: torch.Tensor) -> torch.Tensor:

        batch_size = confounders.size()[0]
        n_sites = gamma_logit.size()[1]

        # gamma is of shape (n_tissues, n_sites)
        # -> reshape it to (1, n_sites, n_tissues)
        gamma_logit = gamma_logit.t().unsqueeze(0).repeat(batch_size, 1, 1)

        # confounders is of shape (batch_size, n_confounders)
        # -> reshape it to (batch_size, 1, n_confounders)
        confounders = confounders.unsqueeze(1).repeat(1, n_sites, 1)

        # gamma is of shape (1, n_sites, n_tissues)
        # confounders is of shape (batch_size, 1, n_confounders)
        # NN input is of shape (batch_size * n_sites, n_confounders + n_tissues)
        input_ = torch.cat((gamma_logit, confounders), 2)

        # NN output is of shape (batch_size * n_sites, n_tissues)
        output_ = 0.1 * self.layers(input_)
        output_ = torch.sigmoid(gamma_logit + output_)
        output_ = torch.nan_to_num(output_)

        # Output is of shape (batch_size, n_tissues, n_sites)
        output_ = torch.transpose(output_.view(batch_size, n_sites, self.n_tissues), 1, 2)

        return output_

    @staticmethod
    def init_weights(m: torch.nn.Module):
        if type(m) in [torch.nn.Conv1d, torch.nn.Linear]:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(np.random.normal(0, 0.005))
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.xavier_uniform_(m.weight)


class Optimizer:

    def __init__(self, alpha_up=1.015, alpha_down=0.9, lr_lb=0, lr_ub=1e+5, verbose=False):
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self.lr_lb = lr_lb
        self.lr_ub = lr_ub
        self.verbose = verbose

        self.optimizers = []
        self.parameters = []
        self.cursor = 0
        self.last_loss = np.inf

    def add(self, param, lr):
        params = [{
            'params': param,
            'lr': lr
        }]
        self.parameters.append(params[0])
        self.optimizers.append(torch.optim.Adam(params, lr=lr))

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self, loss):

        # Update learning rate
        if self.cursor >= len(self.parameters):
            return
        previous_cursor = (self.cursor + len(self.parameters) - 1) % len(self.parameters)
        if loss <= self.last_loss:
            factor = self.alpha_up
        else:
            factor = self.alpha_down
        lr = self.parameters[previous_cursor]['lr']
        lr = float(np.clip(lr * factor, self.lr_lb, self.lr_ub))
        self.parameters[previous_cursor]['lr'] = lr
        for group in self.optimizers[previous_cursor].param_groups:
            group['lr'] = lr
        self.last_loss = loss

        self.optimizers[self.cursor].step()

        self.cursor = (self.cursor + 1) % len(self.parameters)


class MetDecodeV2:

    def __init__(self,
                 R_methylated, R_depths, X_methylated, X_depths,
                 lambda1: float = 2,
                 lambda2: float = 0.0005,
                 lambda3: float = 0.0001,
                 correction: bool = True,
                 n_unknown_tissues=0
                 ):

        self.X_methylated = X_methylated
        self.X_depths = X_depths
        self.R_methylated = R_methylated
        self.R_depths = R_depths
        self.n_known_tissues = self.R_methylated.shape[0]
        self.n_unknown_tissues = n_unknown_tissues
        self.n_tissues = self.n_known_tissues + self.n_unknown_tissues

        # Add pseudo-counts
        self.R_methylated, self.R_depths = MetDecodeV2.add_pseudo_counts(self.R_methylated, self.R_depths)
        self.X_methylated, self.X_depths = MetDecodeV2.add_pseudo_counts(self.X_methylated, self.X_depths)

        R = np.asarray([self.R_methylated, self.R_depths - self.R_methylated])
        self.R = np.transpose(R, (1, 2, 0)).reshape(self.R_depths.shape[0], 2 * self.R_depths.shape[1])
        X = np.asarray([self.X_methylated, self.X_depths - self.X_methylated])
        self.X = np.transpose(X, (1, 2, 0)).reshape(self.X_depths.shape[0], 2 * self.X_depths.shape[1])

        n_confounders = 0
        self.side_information_model = SideInformationModule(self.n_tissues, n_confounders, self.R_depths.shape[1])

        self.lambda1: float = lambda1
        self.lambda2: float = lambda2
        self.lambda3: float = lambda3
        self.correction: bool = correction

        self.info = {
            'loss': []
        }

    @staticmethod
    def add_pseudo_counts(methylated, depths, pc=0.8):
        methylated = np.asarray(methylated, dtype=float)
        depths = np.asarray(depths, dtype=float)
        avg_meth = np.sum(methylated) / np.sum(depths)

        # Compute per-row methylation
        row_methylated = np.sum(methylated, axis=1)
        row_depths = np.sum(depths, axis=1)
        mask = np.logical_or(row_methylated == 0, row_methylated == row_depths)
        row_depths[mask] += pc
        row_methylated[mask] += pc * avg_meth
        row_meth = row_methylated / row_depths

        # Compute per-column methylation
        col_methylated = np.sum(methylated, axis=0)
        col_depths = np.sum(depths, axis=0)
        mask = np.logical_or(col_methylated == 0, col_methylated == col_depths)
        col_depths[mask] += pc
        col_methylated[mask] += pc * avg_meth
        col_meth = col_methylated / col_depths

        # Impute missing values
        meth_prior = np.outer(np.ones(len(row_meth)), col_meth)  # TODO: use survival function of Beta distribution instead
        meth_prior = np.clip(meth_prior, 0.01, 0.99)

        mask = np.logical_or(methylated == 0, methylated == depths)
        methylated[mask] += pc * meth_prior[mask]
        depths[mask] += pc

        assert np.all(methylated > 0)
        assert not np.any(np.isnan(methylated))
        assert not np.any(np.isnan(depths))
        assert not np.any(np.isinf(methylated))
        assert not np.any(np.isinf(depths))
        assert np.all(methylated < depths)

        return methylated, depths

    def _nnls(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        Alpha_hat = []
        for i in range(len(B)):
            x, residuals = scipy.optimize.nnls(A.T, B[i], maxiter=3000)
            if self.n_unknown_tissues > 0:
                x = np.concatenate((x, np.full(self.n_unknown_tissues, 0.2 / self.n_unknown_tissues)))
            Alpha_hat.append(x)
        Alpha_hat = np.asarray(Alpha_hat)
        Alpha_hat /= Alpha_hat.sum(axis=1)[:, np.newaxis]
        return Alpha_hat

    def _init_alpha_and_gamma(self) -> Tuple[np.ndarray, np.ndarray]:
        assert not np.any(self.R_depths <= self.R_methylated)
        A = self.R_methylated / self.R_depths
        B = self.X_methylated / self.X_depths
        Alpha_hat = self._nnls(A, B)
        Alpha_hat += 1e-6
        Alpha_hat /= np.sum(Alpha_hat, axis=1)[:, np.newaxis]
        Alpha_logit = np.log(Alpha_hat)

        if self.n_unknown_tissues > 0:
            extra = np.random.rand(self.n_unknown_tissues, A.shape[1]) * 0.98 + 0.01
            A = np.concatenate((A, extra), axis=0)
        gamma_logit = scipy.special.logit(A)

        return Alpha_logit, gamma_logit

    def _objective(self, alpha_logit: torch.Tensor, gamma_logit: torch.Tensor) -> Any:

        alpha = torch.softmax(alpha_logit, dim=1)

        losses = []

        # assert(not np.any(np.isnan(alpha.data.numpy())))
        # assert(not np.any(np.isnan(torch.mm(Alpha, Gamma).data.numpy())))

        #losses.append(torch.sum(0.5 * ((alpha - mean) / std) ** 2 + torch.log(std * np.sqrt(2 * np.pi))))
        # losses.append(torch.sum(torch.std(alpha, dim=0)))
        #losses.append(-Normal().log_likelihood(alpha, mean, std))
        #losses.append(-self.dirichlet.log_likelihood(concentration, alpha))
        #losses.append(100 * torch.sum(torch.std(alpha, dim=0)))

        #entropy = -torch.sum((alpha + 1e-10) * torch.log(alpha + 1e-10))
        #losses.append(-1000 * entropy)

        # Deviance from known reference profiles
        #losses.append(-self.binomial_ref.log_likelihood(Gamma[:, ::2] / (Gamma[:, ::2] + Gamma[:, 1::2])))
        #losses.append(-self.multinomial_ref.log_likelihood(Gamma))

        #losses.append(-self.multinomial_ref.log_likelihood(gamma_logit))
        if self.correction:
            gamma_logit = self.side_information_model.adapt(gamma_logit)
        gamma = torch.sigmoid(gamma_logit)
        self.gamma_corrected = gamma.cpu().data.numpy()
        #losses.append(-self.binomial_ref.log_likelihood(log_gamma_r))
        weights = torch.FloatTensor(self.R_depths / np.sum(self.R_depths))
        meth = torch.FloatTensor(self.R_methylated / self.R_depths)
        losses.append(torch.sum(weights * ((gamma[:self.n_known_tissues, :] - meth) ** 2)))

        # Maximize proportion of explained counts
        if self.n_unknown_tissues > 0:
            extra = gamma[self.n_known_tissues:, :]
            losses.append(self.lambda1 * torch.mean((extra - torch.median(gamma[:self.n_known_tissues, :], dim=0)[0].unsqueeze(0)) ** 2))
            # losses.append(0.005 * torch.mean(extra * (1. - extra)))
            tmp = torch.sum(alpha[:, self.n_unknown_tissues:], dim=1)
            # tmp = torch.logsumexp(log_alpha[:, self.n_unknown_tissues:], dim=1)
            losses.append(self.lambda2 * torch.mean((tmp - torch.mean(tmp)) ** 2))

        losses.append(self.lambda3 * torch.mean((alpha - torch.mean(alpha, dim=0)) ** 2))

        """
        # Convolution: (batch_size, n_tissues) x (batch_size, n_tissues, n_sites)
        # -> (batch_size, n_sites)
        Gamma_corrected = self.side_information_model.forward(Gamma, self.confounders)

        weights = 1 + 9 * (~self.bisulfite_mask)
        bias = torch.mean(weights.unsqueeze(1) * ((Gamma_corrected - Gamma.unsqueeze(0)) ** 2))
        losses.append(bias)

        if self.is_atlas_comprehensive:
            X_reconstructed = torch.einsum('ij,ijk->ik', alpha, Gamma_corrected)
        else:
            if verbose:
                print(f'Fraction of unexplained counts: {torch.mean(alpha[:, -1]).item()}')
            X_reconstructed = torch.einsum('ij,ijk->ik', alpha[:, :-1], Gamma_corrected)
        """

        # Reconstruction error of patients' profiles
        X_reconstructed = torch.mm(alpha, gamma)
        # log_X_reconstructed = torch.logsumexp(log_alpha.unsqueeze(2) + log_gamma.unsqueeze(0), dim=1)

        #print(torch.min(X_reconstructed), torch.max(X_reconstructed), torch.min(Gamma), torch.max(Gamma))
        #losses.append(-self.multinomial.log_likelihood(X_reconstructed))
        #X_reconstructed = torch.nan_to_num(torch.clamp(X_reconstructed, 1e-5, 1 - 1e-5))
        #losses.append(-self.binomial.log_likelihood(log_X_binary))
        weights = torch.FloatTensor(self.X_depths / np.sum(self.X_depths))
        meth = torch.FloatTensor(self.X_methylated / self.X_depths)
        losses.append(torch.sum(weights * ((X_reconstructed - meth) ** 2)))
        #losses.append(-self.multinomial.log_likelihood(log_X_reconstructed))

        #losses.append(-Dirichlet().log_likelihood(coverage, torch.logaddexp(gamma_logit[:, ::2], gamma_logit[:, 1::2])))

        # Entropy regularization
        #losses.append(1000 * torch.sum(alpha_logit * torch.exp(alpha_logit)))

        # Entropy regularization
        # losses.append(torch.sum(Gamma * torch.log(Gamma)))

        #print(f'Losses: {[loss.item() for loss in losses]}')

        return sum(losses)

    def deconvolute(self,
                     max_n_iter: int = 2000,
                     patience: int = 1000,
                     verbose: bool = True,
                     eps: float = 1e-3) -> np.ndarray:

        self.info['loss'] = []

        _alpha_logit, _gamma_logit = self._init_alpha_and_gamma()
        alpha_logit = torch.nn.Parameter(torch.FloatTensor(np.copy(_alpha_logit)))
        gamma_logit = torch.nn.Parameter(torch.FloatTensor(np.copy(_gamma_logit)))

        optimizer = Optimizer(verbose=False)
        optimizer.add([alpha_logit], 1e-2)
        if self.correction:
            optimizer.add([gamma_logit], 1e-3)
        optimizer.add(list(self.side_information_model.parameters()), 1e-3)

        n_steps_without_improvement = 0
        best_loss = np.inf
        extra_info = {'loss': []}
        for iteration in range(max_n_iter):

            optimizer.zero_grad()
            loss = self._objective(alpha_logit, gamma_logit)
            loss.backward()

            self.info['loss'].append(loss.item())

            #if verbose:
            #    print(f'Loss at iteration {iteration + 1} / {max_n_iter}: {loss.item()}')
            extra_info['loss'].append(loss.item())

            # Update parameters
            #optimizer.step()
            optimizer.step(loss.item())

            # scheduler.step()
            if loss.item() >= best_loss:
                n_steps_without_improvement += 1
                if n_steps_without_improvement >= patience:
                    break
            else:
                best_loss = loss.item()
                n_steps_without_improvement = 0

        self.R_atlas = torch.sigmoid(gamma_logit).cpu().data.numpy()

        print(f'Final loss: {self.info["loss"][-1]}')

        alpha_arr = scipy.special.softmax(alpha_logit.data.numpy(), axis=1)

        alpha_arr = alpha_arr[:, :self.n_known_tissues]
        return alpha_arr
