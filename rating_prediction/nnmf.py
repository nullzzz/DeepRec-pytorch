import torch.nn as nn
import torch
from torch import Tensor


class NNMF(nn.Module):
    def __init__(self, num_user: int, num_item: int, num_factor_1: int = 100, num_factor_2: int = 100,
                 hidden_dimension=50):
        super().__init__()
        self.p = nn.Parameter(torch.normal(size=[num_user, num_factor_1], mean=0, std=0.01))
        self.q = nn.Parameter(torch.normal(size=[num_item, num_factor_1], mean=0, std=0.01))

        self.u = nn.Parameter(torch.normal(size=[num_user, num_factor_2], mean=0, std=0.01))
        self.v = nn.Parameter(torch.normal(size=[num_item, num_factor_2], mean=0, std=0.01))

        self.layer1 = nn.Linear(in_features=2 * num_factor_1 + num_factor_2,
                                out_features=2 * num_factor_1 + num_factor_2, bias=True)

        self.layer2 = nn.Linear(in_features=2 * num_factor_1 + num_factor_2, out_features=hidden_dimension, bias=True)

        self.layer3 = nn.Linear(in_features=hidden_dimension, out_features=hidden_dimension, bias=True)
        self.layer4 = nn.Linear(in_features=hidden_dimension, out_features=hidden_dimension, bias=True)
        self.layer5 = nn.Linear(in_features=hidden_dimension, out_features=1, bias=True)

    def forward(self, user, item):
        inputs = torch.cat([torch.index_select(self.p, 0, user),
                            torch.index_select(self.q, 0, item),
                            torch.multiply(torch.index_select(self.u, 0, user),
                                           torch.index_select(self.v, 0, item))
                            ], dim=1)

        x = self.layer1(inputs)
        x = torch.sigmoid(x)

        x = self.layer2(x)
        x = torch.sigmoid(x)

        x = self.layer3(x)
        x = torch.sigmoid(x)

        x = self.layer4(x)
        x = torch.sigmoid(x)

        x = self.layer5(x)

        return torch.flatten(x)

    def loss_func(self, y: Tensor, predict_value: Tensor, regularization_rate: float = 0.001):
        return torch.sum(torch.square(predict_value - y)) + regularization_rate * (
                torch.norm(self.q) +
                torch.norm(self.p) +
                torch.norm(self.u) +
                torch.norm(self.v))
