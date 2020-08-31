"""
Source: https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py

Implements various flows.

Each flow is invertible so it can be forward()ed and backward()ed.

Notice that backward() is not backward as in backprop but simply inversion.

Each flow also outputs its log det J "regularization"
Reference:

NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
https://arxiv.org/abs/1410.8516

Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
https://arxiv.org/abs/1505.05770

(Laurent's extension of NICE) Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803

(IAF) Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934

(MAF) Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017 
https://arxiv.org/abs/1705.07057
"The advantage of Real NVP compared to MAF and IAF is that it can both generate data and estimate densities with one forward pass only, whereas MAF would need D passes to generate data and IAF would need D passes to estimate densities."

Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, Jul 2018
https://arxiv.org/abs/1807.03039

(Review paper) Normalizing Flows for Probabilistic Modeling and Inference
https://arxiv.org/abs/1912.02762
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nflib.nets import LeafParam, MLP, ARMLP, SpectralNormMLP
from nflib.utils import power_series_matrix_logarithm_trace


class AffineConstantFlow(nn.Module):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is 0
    """

    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = (
            nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        )
        self.t = (
            nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None
        )

    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization, where on the very first batch we clever initialize (s, t) so that the output is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False

    def forward(self, x):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None  # for now
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().forward(x)


class AffineHalfFlow(nn.Module):
    """
     As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, dim, parity=False, net_class=MLP, nh=24, scale=True, shift=True):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = net_class(self.dim // 2, self.dim // 2, nh)
        if shift:
            self.t_cond = net_class(self.dim // 2, self.dim // 2, nh)

    def forward(self, x):
        x0, x1 = x[:, ::2], x[:, 1::2]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0)
        t = self.t_cond(x0)
        z0 = x0  # untouched half
        z1 = torch.exp(s) * x1 + t  # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.zeros((x.shape[0], self.dim))
        z[:, ::2] = z0
        z[:, 1::2] = z1
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z):
        z0, z1 = z[:, ::2], z[:, 1::2]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0)
        t = self.t_cond(z0)
        x0 = z0  # untouched half
        x1 = (z1 - t) * torch.exp(-s)  # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.zeros((z.shape[0], self.dim))
        x[:, ::2] = x0
        x[:, 1::2] = x1
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        xs = [z]
        for flow in self.flows:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior, flows, compute_grad=False):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)
        self.compute_grad = compute_grad

    def forward(self, x):
        if self.compute_grad:
            x = torch.tensor(x, requires_grad=True)
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z):
        if self.compute_grad:
            z = torch.tensor(z, requires_grad=True)
        xs, log_det = self.flow.backward(z)
        return xs, log_det

    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        if self.compute_grad:
            z = torch.tensor(z, requires_grad=True)
        xs, _ = self.flow.backward(z)
        return xs


class ContractiveResidualFlow(nn.Module):
    """ 
    If a neural network g_res is contractive, the residual transformation is guaranteed to be invertible;
    Spectral Normalization (https://arxiv.org/abs/1802.05957) is used to enforce contractiveness;
    Hutchinson's trace estimator is used for fast computation of Jacobian determinant
    """

    def __init__(
        self,
        dim,
        net_class=SpectralNormMLP,
        nh=24,
        inv_iters=100,
        coeff=0.97,
        n_power_iter=5,
        num_trace_samples=1,
        num_series_terms=1,
    ):
        super().__init__()
        self.dim = dim
        self.g_res = net_class(
            self.dim, self.dim, nh, coeff=coeff, n_power_iter=n_power_iter
        )
        self.inv_iters = inv_iters
        self.num_trace_samples = num_trace_samples
        self.num_series_terms = num_series_terms

    def forward(self, x):
        gx = self.g_res(x)
        z = x + gx
        log_det = power_series_matrix_logarithm_trace(
            gx, x, self.num_series_terms, self.num_trace_samples
        )
        return z, log_det

    def backward(self, z):
        zk = z
        for i in range(self.inv_iters):
            gz = self.g_res(zk)
            x = zk - gz
            zk = x
        log_det = power_series_matrix_logarithm_trace(
            gz, z, self.num_series_terms, self.num_trace_samples
        )
        return x, log_det
