"""
Various helper network modules
Sources: 
- https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/nets.py
- https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/made.py
"""

from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, nin, nout, bias=True):
        super().__init__(nin, nout, bias)
        self.register_buffer("mask", torch.ones(nout, nin))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class LeafParam(nn.Module):
    """
    ignores the input and outputs a parameter tensor
    """

    def __init__(self, n):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1, n))

    def forward(self, x):
        return self.p.expand(x.size(0), self.p.size(1))


class PositionalEncoder(nn.Module):
    """
    Each dimension of the input gets expanded out with sins/coses
    to "carve" out the space. Useful in low-dimensional cases
    with tightly "curled up" data
    More details here: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    """

    def __init__(self, freq=(0.5, 1, 2, 4, 8)):
        super().__init__()
        self.freqs = freqs

    def forward(self, x):
        sines = [torch.sin(x * f) for f in self.freqs]
        coses = [torch.cos(x * f) for f in self.freqs]
        out = torch.cat(sines + coses, dim=1)
        return out


class MLP(nn.Module):
    """ a simple `nlayers`-layer MLP """

    def __init__(self, nin, nout, nh, nlayers=4, neg_slope=0.2):
        super().__init__()
        assert nlayers > 1, "nlayers must be > 1"
        layers = [
            ("input", nn.Linear(nin, nh)),
            ("leakyRelu0", nn.LeakyReLU(neg_slope)),
        ]
        for i in range(nlayers - 2):
            layers += [
                ("linear{}".format(i + 1), nn.Linear(nh, nh)),
                ("leakyRelu{}".format(i + 1), nn.LeakyReLU(neg_slope)),
            ]
        layers += [
            ("output", nn.Linear(nh, nout)),
        ]
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.net(x)


class PosEncMLP(nn.Module):
    """
    Position Encoded MLP, where the first layer performs position encoding.
    Each dimension of the input gets transformed to len(freqs)*2 dimensions using a fixed transformation of sin/cos of given frequencies.
    """

    def __init__(
        self, nin, nout, nh, freqs=(0.5, 1, 2, 4, 8), nlayers=4, neg_slope=0.2
    ):
        super().__init__()
        self.net = nn.Sequential(
            PositionalEncoder(freqs),
            MLP(nin * len(freqs) * 2, nout, nh, nlayers=nlayers, neg_slope=neg_slope),
        )

    def forward(self, x):
        return self.net(x)


class MADE(nn.Module):
    """ 
        Masked AutoEncoder for Density Estimation, Germain et al., 2015
        https://arxiv.org/abs/1502.03509
    """

    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False):
        """ 
            nin: integer; number of inputs
            hidden sizes: a list of integers; number of units in hidden layers
            nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
                note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
                will be all the means and the second nin will be stds. i.e. output dimensions depend on the
                same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
                the output of running the tests for this file makes this a bit more clear with examples.
            num_masks: can be used to train ensemble over orderings/connections
            natural_ordering: force natural ordering of dimensions, don't use random permutations
        """
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [MaskedLinear(h0, h1), nn.ReLU(),]
            )
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling throught num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this could get memory expensive for large number of masks

    def update_masks(self):
        if self.m and self.num_masks == 1:
            return  # only a single seed, skip for efficiency

        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = (
            np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        )
        for l in range(L):
            self.m[l] = rng.randint(
                self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l]
            )

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.net(x)


class ARMLP(nn.Module):
    """ a `nlayers`-layer auto-regressive MLP, wrapper around MADE net """

    def __init__(self, nin, nout, nh, nlayers=3):
        super().__init__()
        self.net = MADE(nin, nlayers * [nh], nout, num_masks=1, natural_ordering=True)

    def forward(self, x):
        return self.net(x)
