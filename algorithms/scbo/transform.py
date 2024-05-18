import torch
from torch import Tensor


def gaussian_copula(Y: Tensor, maximize: bool = True) -> Tensor:
    order = torch.argsort(Y, descending=maximize)
    num_data = Y.numel()
    empi_cdf = (torch.arange(len(Y))+1) / (num_data+1)
    Y_new = torch.distributions.Normal(0,1).icdf(empi_cdf)
    Y_new[order] = Y_new.clone()
    if maximize:
        Y_new = -Y_new
    return Y_new


def bilog(Y: Tensor) -> Tensor:
    Y_new = Y.float()
    Y_new[Y_new == 0] = -1
    Y_new = torch.sgn(Y_new) * torch.log(1 + torch.abs(Y_new))
    return Y_new
