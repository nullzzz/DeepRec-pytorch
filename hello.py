import argparse

import torch
from torch.optim import RMSprop, Adam
from tqdm import tqdm, trange

from func import *
from load_data_rating import load_data_rating
from rating_prediction.mf import MF
from rating_prediction.nnmf import NNMF


def parse_args():
    parser = argparse.ArgumentParser(description="DeepRec-pytorch")
    parser.add_argument("--model", choices=["MF", "NNMF"],
                        default="MF")
    parser.add_argument("--dataset", type=str, default="./data/AllDataList.csv")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--regularization_rate", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_name = args.model
    dataset = args.dataset
    device_name = args.device
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    regularization_rate = args.regularization_rate

    train_data, test_data, n_users, n_items = load_data_rating(dataset)

    device = torch.device(device_name)
    if model_name == "NNMF":
        model = NNMF(n_users, n_items)
        optimizer = RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.001)
    elif model_name == "MF":
        model = MF(n_users, n_items)
        optimizer = Adam(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError
    model = model.to(device)

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

    bar = trange(epochs, bar_format="Epoch {n_fmt}/{total_fmt}: |{bar}| [{elapsed}<{remaining} {rate_fmt}] {postfix}", ncols=152)
    num_training = len(rating)
    total_batch = num_training // batch_size
    for epoch in bar:
        model.train()
        bar.set_description(f"Epoch {epoch}/{epochs}")
        # shuffled ordering
        index = torch.tensor(np.random.permutation(num_training), device=device, dtype=torch.long)
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

            # if bar:
            #     bar.set_postfix(loss=loss.item())

        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                pred = model(test_user, test_item)
                pred = pred.cpu().numpy()

                bar.set_postfix(dict(
                    epoch=epoch,
                    MAE=mae(pred, test_rating),
                    RMSE=rmse(pred, test_rating),
                    MSE=mse(pred, test_rating)))
