import torch.nn as nn
import torch
from torch import Tensor
from torch.optim import RMSprop
from tqdm import tqdm

from load_data_rating import load_data_rating
from func import *


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


if __name__ == "__main__":
    train_data, test_data, n_users, n_items = load_data_rating("../data/AllDataList.csv")

    lr = 0.001
    epochs = 500
    batch_size = 1024
    device = torch.device("cuda:1")
    model = NNMF(n_users, n_items)
    model = model.to(device)

    loss_fn = torch.nn.MSELoss(reduction='sum')

    optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=0.001)

    print(len(list(model.parameters())))
    t = train_data.tocoo()
    user = t.row
    item = t.col
    rating = t.data

    t = test_data.tocoo()
    test_user = torch.tensor(t.row, device=device)
    test_item = torch.tensor(t.col, device=device)
    test_rating = t.data

    user = torch.tensor(user, device=device)
    item = torch.tensor(item, device=device)
    rating = torch.tensor(rating, device=device, dtype=torch.float)

    bar = tqdm(range(epochs))
    num_training = len(rating)
    total_batch = num_training // batch_size
    for epoch in bar:
        model.train()
        bar.set_description(f"Epoch {epoch}/{epochs}")

        index = torch.tensor(np.random.permutation(num_training), device=device)  # shuffled ordering

        user_random = user[index]
        item_random = item[index]
        rating_random = rating[index]

        # train
        for i in range(total_batch):

            batch_user = user_random[i * batch_size:(i + 1) * batch_size]
            batch_item = item_random[i * batch_size:(i + 1) * batch_size]
            batch_rating = rating_random[i * batch_size:(i + 1) * batch_size]

            pred = model(batch_user, batch_item)

            loss = model.loss_func(pred, batch_rating)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if bar:
                bar.set_postfix(loss=loss.item())

        if epoch % 10 == 9:
            with torch.no_grad():
                model.eval()
                pred = model(test_user, test_item)
                pred = pred.cpu().numpy()

                bar.set_postfix(epoch=epoch, MAE=mae(pred, test_rating),
                                RMSE=rmse(pred, test_rating),
                                MSE=mse(pred, test_rating))
