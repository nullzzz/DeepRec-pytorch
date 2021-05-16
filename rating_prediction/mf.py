import torch
import torch.nn as nn
from torch import Tensor


class MF(nn.Module):
    def __init__(self, num_user: int, num_item: int, num_factor: int = 30):
        super(MF, self).__init__()
        # n * k
        self.p = nn.Parameter(torch.normal(size=[num_user, num_factor], mean=0, std=0.01))
        self.q = nn.Parameter(torch.normal(size=[num_item, num_factor], mean=0, std=0.01))

        # bias of user and item
        self.b_u = nn.Parameter(torch.normal(size=[num_user], mean=0, std=0.01))
        self.b_i = nn.Parameter(torch.normal(size=[num_item], mean=0, std=0.01))

    def forward(self, user, item):
        user_latent_factor = torch.index_select(self.p, 0, user)
        item_latent_factor = torch.index_select(self.q, 0, item)

        user_bias = torch.index_select(self.b_u, 0, user)
        item_bias = torch.index_select(self.b_i, 0, item)

        return torch.sum(torch.multiply(user_latent_factor, item_latent_factor), 1) + user_bias + item_bias

    def loss_func(self, y: Tensor, predict_value: Tensor, regularization_rate: float = 0.001):
        return torch.sum(torch.square(y - predict_value)) + \
               regularization_rate * (
                       torch.norm(self.b_i) +
                       torch.norm(self.b_u) +
                       torch.norm(self.p) +
                       torch.norm(self.q))

