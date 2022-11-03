import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import context

import numpy as np
import time
import matplotlib.pyplot as plt
from math import pi
from scipy.interpolate import griddata

from src.model import *
from src.dataset import Trainset_poisson
from src.config import Options_poisson
from src.eager_lbfgs import lbfgs, Struct

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")



def train():
    args = Options_poisson().parse()
    trainset = Trainset_poisson(args.n_x, args.n_y, args.n_b)
    args.trainset = trainset
    xy, xy_b, u_b = trainset()

    global data_length
    data_length = xy.shape[0]

    # 实例化前向网络
    model = Modified_MLP(2, 1, dim_hidden=args.dim_hidden, hidden_layers=args.hidden_layers)

    # 设定损失函数并连接前向网络与损失函数(PINN)
    Get_Loss = PINN_poisson(model)
    Get_Loss.to_float(ms.float16)

    # 设定优化器
    lr = nn.exponential_decay_lr(args.lr, args.decay_rate, args.epochs_Adam, args.step_per_epoch, args.decay_steps)
    optimizer_Adam = nn.Adam(params=Get_Loss.trainable_params(), learning_rate=lr)

    # 定义训练网络
    train_net = CustomTrainOneStepCell(Get_Loss, optimizer_Adam)

    # 设置网络为训练模式
    train_net.set_train();

    loss_list = []
    train_info = "train_info.txt"
    open("train_info.txt", 'w').close()
    start = time.time()
    for epoch in range(args.epochs_Adam):
        loss_value = train_net(xy, xy_b, u_b)
        loss_list.append(loss_value.asnumpy().item())

        if (epoch + 1) % 100 == 0:
            running_time = time.time() - start
            start = time.time()
            info = f'Epoch #  {epoch + 1}   loss:{loss_value.asnumpy().item():.2e}   time:{running_time:.2f}'
            print(info)

            with open(train_info, "a+") as f:
                f.write(info + '\n')
                f.close

    # 保存Adam模型
    ms.save_checkpoint(model, "model_adam.ckpt")

    # # 导入Adam模型
    param_dict = ms.load_checkpoint("model_adam.ckpt");
    ms.load_param_into_net(model, param_dict);

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    train_net_lbfgs = CustomTrainOneStepCell_lbfgs(Get_Loss, optimizer_Adam)

    sizes = []
    for param in model.get_parameters():
        if len(param.shape) == 2:
            size = param.shape[0] * param.shape[1]
        else:
            size = param.shape[0]
        sizes.append(size)

    indexs = [0]
    for i in range(len(sizes)):
        index = sum(sizes[:i + 1])
        indexs.append(index)


    # 将网络的parameter拿出至列表
    def get_weights(net):
        """ Extract parameters from net, and return a list of tensors"""
        w = []
        for p in net.get_parameters():
            w.extend(p.asnumpy().flatten())

        w = ms.Tensor(w).astype('float16')
        return w


    # 将列表放回到网络的parameter
    def set_weights(model, weights, indexs):
        for (i, p) in enumerate(model.get_parameters()):
            if p.requires_grad == True:
                w = weights[indexs[i]: indexs[i + 1]]
                w = w.reshape(p.shape)
                p.set_data(w, ms.float16)


    def get_loss_and_flat_grad(xy, xy_b, u_b):
        def loss_and_flat_grad(weights):
            set_weights(Get_Loss, weights, indexs)
            loss_value = train_net_lbfgs(xy, xy_b, u_b)
            grads = train_net_lbfgs.grad(Get_Loss, train_net_lbfgs.weights)(xy, xy_b, u_b)

            grad_flat = []
            for g in grads:
                grad_flat.append(g.reshape([-1]))

            grad_flat = ms.numpy.concatenate(grad_flat)
            return loss_value, grad_flat

        return loss_and_flat_grad

    Get_Loss.to_float(ms.float32)
    param_list = get_weights(model)
    loss_and_flat_grad =get_loss_and_flat_grad(xy, xy_b, u_b)
    newton_iter = args.epochs_LBFGS

    lbfgs(loss_and_flat_grad,
          param_list,
          Struct(), maxIter=newton_iter, learningRate=1)

    # 保存LBFGS模型
    ms.save_checkpoint(model, "model_lbfgs.ckpt")

    # 导入LBFGS模型
    param_dict = ms.load_checkpoint("model_lbfgs.ckpt");
    ms.load_param_into_net(model, param_dict);


    ## 测试
    def predict(x, y):
        x = ms.Tensor(x, dtype=ms.float16)
        y = ms.Tensor(y, dtype=ms.float16)

        u_star = model(x, y)

        return u_star.asnumpy()


    def exact_sol(x, y):
        sol = np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y) / (32 * np.pi ** 2)
        return sol


    # 设置网络为测试模式
    train_net.set_train(False);

    # 计算内部点error
    nx, ny = (201, 201)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    x, y = np.meshgrid(x, y)
    xy = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    xy = xy[(xy[:, 1] - xy[:, 0] - 0.5 < 0) * (xy[:, 1] + xy[:, 0] - 1.5 < 0)]

    u_star = exact_sol(xy[:, [0]], xy[:, [1]])
    u_pred = predict(xy[:, [0]], xy[:, [1]])

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    info = '内部点的Error u: %e' % (error_u)
    print(info)
    with open(train_info, "a+") as f:
        f.write(info + '\n')
        f.close

    # 计算边界点error
    n_b = 256
    y1 = np.linspace(0, 0.5, n_b)
    x1 = 0 * y1
    xy_b1 = np.concatenate([x1.reshape(-1, 1), y1.reshape(-1, 1)], axis=1)

    x2 = np.linspace(0, 0.5, n_b)
    y2 = x2 + 0.5
    xy_b2 = np.concatenate([x2.reshape(-1, 1), y2.reshape(-1, 1)], axis=1)

    x3 = np.linspace(0.5, 1, n_b)
    y3 = -x3 + 1.5
    xy_b3 = np.concatenate([x3.reshape(-1, 1), y3.reshape(-1, 1)], axis=1)

    y4 = np.linspace(0, 0.5, n_b)
    x4 = 0 * y4 + 1
    xy_b4 = np.concatenate([x4.reshape(-1, 1), y4.reshape(-1, 1)], axis=1)

    x5 = np.linspace(0, 1, n_b)
    y5 = 0 * x5
    xy_b5 = np.concatenate([x5.reshape(-1, 1), y5.reshape(-1, 1)], axis=1)

    xy_b = np.vstack([xy_b1, xy_b2, xy_b3, xy_b4, xy_b5])

    u_star = exact_sol(xy_b[:, [0]], xy_b[:, [1]])
    u_pred = predict(xy_b[:, [0]], xy_b[:, [1]])

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    info = '边界点的Error u: %e' % (error_u)
    print(info)
    with open(train_info, "a+") as f:
        f.write(info + '\n')
        f.close

    # %%

    nx, ny = (201, 201)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    xv, yv = np.meshgrid(x, y)
    Exact_u = exact_sol(xv, yv)

    x = np.reshape(x, (-1, 1))
    y = np.reshape(y, (-1, 1))

    X, Y = np.meshgrid(x, y)
    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    u_star = Exact_u.flatten()[:, None]

    u_pred = predict(X.flatten()[:, None], Y.flatten()[:, None])

    U_pred = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')

    for i in range(Exact_u.shape[0]):
        for j in range(Exact_u.shape[1]):
            if ~((yv[i, j] - xv[i, j] - 0.5 < 0) * (yv[i, j] + xv[i, j] - 1.5 < 0)):
                Exact_u[i, j] = np.nan
                U_pred[i, j] = np.nan

    # %%

    plt.rcParams.update({'font.size': 18})

    fig = plt.figure(3, figsize=(18, 5))
    ax = plt.subplot(1, 3, 1)
    plt.pcolor(xv, yv, Exact_u, cmap='jet')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(r'Exact $u(x,y)$')
    plt.tight_layout()
    ax.set_aspect(1. / ax.get_data_ratio())

    ax = plt.subplot(1, 3, 2)
    plt.pcolor(xv, yv, U_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(r'Predicted $u(x,y)$')
    plt.tight_layout()
    ax.set_aspect(1. / ax.get_data_ratio())

    ax = plt.subplot(1, 3, 3)
    plt.pcolor(xv, yv, np.abs(Exact_u - U_pred), cmap='jet')
    cbar = plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(r'Absolute error')
    plt.tight_layout()
    ax.set_aspect(1. / ax.get_data_ratio())

    plt.savefig('result.png')
    plt.show()

if __name__ == '__main__':
    train()