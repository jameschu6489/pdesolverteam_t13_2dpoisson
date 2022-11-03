import mindspore as ms
import numpy as np


class Trainset_poisson():
    def __init__(self, *args):
        self.args = args

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
        xy = xy[(xy[:, 0] - 0.5) ** 2 + (xy[:, 1] - 0.5) ** 2 <= 0.25]

        # 边界点
        theta_bcs = np.linspace(0, np.pi * 2, n_b)
        r_bcs = np.array([0.5])
        theta_bcs, r_bcs = np.meshgrid(theta_bcs, r_bcs)
        xx_bcs = r_bcs * np.cos(theta_bcs) + 0.5
        yy_bcs = r_bcs * np.sin(theta_bcs) + 0.5
        xy_b = np.concatenate((xx_bcs.reshape((-1, 1)), yy_bcs.reshape((-1, 1))), axis=1)
        u_b = np.sin(4*np.pi * xy_b[:, [0]]) * np.sin(4*np.pi * xy_b[:, [1]]) / (32 * np.pi ** 2)

        xy = ms.Tensor(xy, dtype=ms.float32)
        xy_b = ms.Tensor(xy_b, dtype=ms.float32)
        u_b = ms.Tensor(u_b, dtype=ms.float32)
        return xy, xy_b, u_b

