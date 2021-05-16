import numpy as np


def shuffle(data1, data2):
    state = np.random.get_state()
    np.random.shuffle(data1)

    np.random.set_state(state)
    np.random.shuffle(data2)


def rmse(prediction: np.ndarray, rating: np.ndarray):
    return np.sqrt(np.mean(np.power(prediction - rating, 2)))


def mse(prediction: np.ndarray, rating: np.ndarray):
    return np.mean(np.power(prediction - rating, 2))


def mae(prediction: np.ndarray, rating: np.ndarray):
    return np.mean(np.abs(prediction - rating))
