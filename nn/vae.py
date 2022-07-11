import torch as th
from torch import nn
from torch.autograd import Function
from typing import List, Tuple


class KLLossFunction(Function):
    @staticmethod
    def forward(ctx, mu: th.Tensor, logsigma: th.Tensor, lr=1.0):
        ctx.save_for_backward(mu, logsigma, th.tensor(lr, device=mu.device))
        return mu, logsigma

    @staticmethod
    def backward(ctx, grad_mu: th.Tensor, grad_sigma: th.Tensor):
        mu, logsigma, lr = ctx.saved_tensors

        # KL Divergence for the grad
        #kl_div = 0.5 * (th.exp(logsigma) + mu**2 - 1 - logsigma).sum(dim=1).mean()
        
        # Undoes grad.sum(dim=1).mean()
        n_mean = mu.numel() / mu.shape[1]

        kl_mu    = mu / n_mean
        kl_sigma = 0.5 * (th.exp(logsigma) - 1) / n_mean

        grad_mu = grad_mu + lr * kl_mu
        grad_sigma = grad_sigma + lr * kl_sigma
        return grad_mu, grad_sigma, None

class VariationalFunction(nn.Module):
    """
    Variational Layer with KLDiv Loss, samples from a given predicted mu and sigma distribution;
    Needs an additional Encoder and Decoder for an VAE
    """
    def __init__(self, mean=0, factor=1, groups=1):
        """
        Init
        :param factor: kl gradient scalling factor
        """
        super(VariationalFunction, self).__init__()
        self.factor = factor
        self.mean = mean
        self.groups = groups

        # Kullback Leibler Divergence
        self.kl = KLLossFunction.apply

    def forward(self, input: th.Tensor):
        # Encodes latent state
        input = input.view(input.shape[0], self.groups, -1, *input.shape[2:])
        mu, logsigma = th.chunk(input, chunks=2, dim=2)

        # Adds gradients from Kullback Leibler Divergence loss
        mu, logsigma = self.kl(th.clip(mu, -100, 100), th.clip(logsigma, -1000, 10), self.factor)

        # Sampled from the latent state
        noise = th.normal(mean=self.mean, std=1, size=logsigma.shape, device=logsigma.device)
        z = mu + (th.exp(0.5 * logsigma) * noise if self.training else 0) # TODO verifiy training / testing noise!!!!!! TODO

        return z.view(z.shape[0], -1, *z.shape[3:])
