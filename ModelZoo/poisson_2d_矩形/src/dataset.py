import mindspore as ms
import numpy as np


class Trainset_poisson():
    def __init__(self, *args):
        self.args = args
        self.shape = (self.args[1], self.args[0])

    def __call__(self):
        return self.data()

    def data(self):
        n_x = self.args[0]
        n_y = self.args[1]
        n_b = self.args[2]

        # 内部点
        x = np.linspace(0, 1, n_x)
        y = np.linspace(0, 1, n_y)
        x, y = np.meshgrid(x, y)
        xy = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

        # 边界
        bound = np.linspace(0, 1, n_b)
        bound = bound.reshape(-1, 1)

        xy_lb = np.hstack((bound, bound * 0))
        xy_ub = np.hstack((bound, bound * 0 + 1))
        xy_left = np.hstack((bound * 0, bound))
        xy_right = np.hstack((bound * 0 + 1, bound))
        xy_b = np.vstack([xy_lb, xy_ub, xy_left, xy_right])

        xy = ms.Tensor(xy, dtype=ms.float32)
        xy_b = ms.Tensor(xy_b, dtype=ms.float32)

        return xy, xy_b