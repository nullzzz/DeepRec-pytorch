import pandas as pd

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


def load_data_rating(path: str, header=('user_id', 'item_id', 'rating', 'category'),
                     test_size=0.1, sep="\t"):
    """
    Loading the data for rating prediction task
    :param path: the path of the dataset, datasets should be in the CSV format
    :param header: the header of the CSV format, the first three should be: user_id, item_id, rating
    :param test_size: the test ratio, default 0.1
    :param sep: the separator for csv columns, default space
    :return:
    """

    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    # number of users and items
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    # split train and test data
    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row: list = []
    train_col: list = []
    train_rating: list = []

    for line in train_data.itertuples():
        u: int = line[1] - 0
        i: int = line[2] - 0
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])

    # train_row[k], train_col[k] -> train_rating[k]
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 0)
        test_col.append(line[2] - 0)
        test_rating.append(line[3])
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))
    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_matrix.todok(), n_users, n_items

