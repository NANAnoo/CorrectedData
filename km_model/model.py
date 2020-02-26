import torch
import torch.utils.data
import numpy as np
import sys
sys.path.append('../')
import km_model.info as info

from torch import nn
from time import time
from tqdm import tqdm
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import PermuteRandom, F_fully_connected, GLOWCouplingBlock
from km_model.utils import MMD_multiscale, fit, non_nagative_attachment,plot_losses

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 网络模型结构
# TODO,维度的选择，网络深度的选择，网络中隐藏层节点的选择，参数的选择,dropout
def model(dim_x, dim_y, dim_z, dim_total, lr, l2_reg, meta_epoch, gamma, hidden_depth=8):
    nodes = []
    # 定义输入层节点
    nodes.append(InputNode(dim_total, name='input'))

    # 定义隐藏层节点
    for k in range(hidden_depth):
        nodes.append(Node(nodes[-1], GLOWCouplingBlock,
                          {'subnet_constructor': F_fully_connected, 'clamp': 2.0, },
                          name='coupling_{k}'))
        nodes.append(Node(nodes[-1], PermuteRandom, {'seed': 1}, name='permute_{k}'))
    nodes.append(Node(nodes[-1], GLOWCouplingBlock,
                      {'subnet_constructor': F_fully_connected, 'clamp': 2.0, },
                      name='coupling_{k}'))
    # 定义输出层节点
    nodes.append(OutputNode(nodes[-1], name='output'))

    # 构建可逆网络
    inn = ReversibleGraphNet(nodes)

    # 定义优化器
    # TODO:参数调整
    optimizer = torch.optim.Adam(inn.parameters(), lr=lr, betas=(0.9, 0.999),
                                 eps=1e-04, weight_decay=l2_reg)

    # 学习率调整
    # TODO:参数调整
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=meta_epoch, gamma=gamma)

    # 损失函数设置
    # x，z无监督：MMD，y有监督：平方误差
    loss_backward = MMD_multiscale
    loss_latent = MMD_multiscale
    loss_fit = fit

    return inn, optimizer, scheduler, loss_backward, loss_latent, loss_fit


# 网络训练过程
def train(model, train_loader, n_its_per_epoch, zeros_noise_scale, batch_size, ndim_tot, ndim_x, ndim_y, ndim_z,
          y_noise_scale, optimizer, lambd_predict, loss_fit, lambd_latent, loss_latent, lambd_rev, loss_backward,
          i_epoch=0):
    model.train()

    l_tot = 0
    batch_idx = 0

    # 训练轮数相关的权重 4-1
    loss_factor = 600 ** (float(i_epoch) / 300) / 600
    if loss_factor > 1:
        loss_factor = 1

    # zeros_noise_scale *= (1 - loss_factor)

    for x, y in train_loader:
        batch_idx += 1
        if batch_idx > n_its_per_epoch:
            break

        x, y = x.to(device), y.to(device)

        y_clean = y.clone()

        # 对x进行向量补齐
        pad_x = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                ndim_x, device=device)
        # 对yz进行向量补齐
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)

        y += y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)
        # add_info += y_noise_scale * torch.randn(batch_size, info_dim, dtype=torch.float, device=device)

        x, y = (torch.cat((x, pad_x), dim=1),
                torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y),
                          dim=1))

        optimizer.zero_grad()

        # 前向训练：
        output = model(x)

        # Shorten output, and remove gradients wrt y, for latent loss_dir
        y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)

        l = lambd_predict * loss_fit(output[:, ndim_z:], y[:, ndim_z:])

        output_block_grad = torch.cat((output[:, :ndim_z],
                                       output[:, -ndim_y:].data), dim=1)

        l += lambd_latent * loss_latent(output_block_grad, y_short)
        l_tot += l.data.item()

        l.backward()

        # Backward step:
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)
        y = y_clean + y_noise_scale * torch.randn(batch_size, ndim_y, device=device)

        orig_z_perturbed = (output.data[:, :ndim_z] + y_noise_scale *
                            torch.randn(batch_size, ndim_z, device=device))
        y_rev = torch.cat((orig_z_perturbed, pad_yz, y), dim=1)
        y_rev_rand = torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y), dim=1)

        output_rev = model(y_rev, rev=True)
        output_rev_rand = model(y_rev_rand, rev=True)

        l_rev = (
                lambd_rev
                * loss_factor
                * (loss_backward(output_rev_rand[:, :ndim_x], x[:, :ndim_x])
                   # + loss_fit(output_rev_rand[:, -info_dim:], x[:, -info_dim:])
                   )
        )

        l_rev += (0.50 * lambd_predict * loss_fit(output_rev, x)
                  + loss_factor * non_nagative_attachment(10, 2, output_rev[:, :ndim_x]).sum())

        l_tot += l_rev.data.item()
        l_rev.backward()

        for p in model.parameters():
            p.grad.data.clamp_(-5.00, 5.00)

        optimizer.step()

    return l_tot / batch_idx, l / batch_idx, l_rev / batch_idx


# 训练模型
def main():
    # 文件名字
    loss_fig_name = 'loss_fig_3'
    loss_txt_name = 'loss_dir/loss_txt_3.txt'
    model_name = 'model_dir/model_03'

    # 训练轮数
    n_epochs = 3000

    # x,y,z的维度
    dim_x = info.base_color_num
    dim_y = info.reflectance_dim
    dim_z = 13
    dim_total = max(dim_x, dim_y + dim_z)
    # 学习率调整
    lr = 1.5e-3  # 初始学习率
    l2_reg = 2e-5  # 权重衰减（L2惩罚）
    meta_epoch = 12  # 调整学习率的步长
    gamma = 0.004 ** (1. / 1333)  # 学习率下降的乘数因子
    # 训练批次及大小
    n_its_per_epoch = 12  # 每次训练12批数据
    batch_size = 1600  # 每批训练1600个样本
    # 用于维度补齐的值
    y_noise_scale = 3e-5
    zeros_noise_scale = 3e-5
    # 损失的权重
    lambd_predict = 200.  # forward pass
    lambd_latent = 300.  # latent space
    lambd_rev = 500.  # backwards pass
    # 其它参数
    test_split = 3 * 3  # 用于测试的数据个数

    print()
    print('---------------------HYPERPARAMETER-------------------')
    print(' n_epochs      :',n_epochs)
    print(' lambd_predict :',lambd_predict)
    print(' lambd_latent  :',lambd_latent)
    print(' lambd_rev     :',lambd_rev)
    print(' learning-rate :',lr)
    print(' gamma         :',gamma)
    print(' l2_reg        :',l2_reg)
    print('------------------------------------------------------')
    print()
    # 获取模型结构
    inn, optimizer, scheduler, loss_backward, loss_latent, loss_fit = model(dim_x, dim_y, dim_z, dim_total,
                                                                            lr=lr, l2_reg=l2_reg, meta_epoch=meta_epoch,
                                                                            gamma=gamma)
    # 读取数据集
    data = np.load('dataset/data_02.npz')
    concentrations = torch.from_numpy(data['concentrations']).float()
    reflectance = torch.from_numpy(data['reflectance']).float()
    # 加载数据
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(concentrations[test_split:], reflectance[test_split:]),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    # 初始化网络权重
    for model_list in inn.children():
        for block in model_list.children():
            for coeff in block.children():
                coeff.fc3.weight.data = 0.01 * torch.randn(coeff.fc3.weight.shape)
    inn.to(device)

    try:
        t_start = time()  # 训练开始时间
        loss_for_list = []  # 记录前向训练的损失
        loss_rev_list = []  # 记录反向训练的损失

        # n_epochs次迭代训练过程
        for i_epoch in tqdm(range(n_epochs), ascii=True, ncols=80):
            scheduler.step()

            # 训练模型，返回损失
            avg_loss, loss_for, loss_rev = train(inn, train_loader, n_its_per_epoch, zeros_noise_scale, batch_size,
                                                 dim_total, dim_x, dim_y, dim_z, y_noise_scale, optimizer,
                                                 lambd_predict, loss_fit, lambd_latent, loss_latent, lambd_rev,
                                                 loss_backward,
                                                 i_epoch)

            # 添加正向和逆向的损失到列表中
            loss_for_list.append(loss_for.item())
            loss_rev_list.append(loss_rev.item())
        # 保存模型
        torch.save(inn, model_name)
        # 损失函数画图
        losses=[loss_for_list,loss_rev_list]
        plot_losses(losses,loss_fig_name)
        # 保存损失函数到文件
        loss_txt = open(loss_txt_name, 'w+')
        for i in range(n_epochs):
            # 打印损失
            print('epoch:', i, ' loss_for:', loss_for_list[i], ' loss_rev:', loss_rev_list[i],
                  file=loss_txt)
        loss_txt.close()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nTraining took %.2f minutes\n" % ((time() - t_start) / 60))


main()
