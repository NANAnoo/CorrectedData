import torch
import numpy as np
import km_model.info as info
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from colormath.color_objects import SpectralColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# -----------------------------------------------光学计算--------------------------------------------------
# 将400-700范围内的reflectance转化为SpectralColor对象
def reflectance_to_spectral_color(reflectance, observer='10', illuminant='d65', start=400, end=710):
    spc = SpectralColor(
        observer=observer, illuminant=illuminant,
        spec_400nm=reflectance[0], spec_410nm=reflectance[1], spec_420nm=reflectance[2],
        spec_430nm=reflectance[3], spec_440nm=reflectance[4], spec_450nm=reflectance[5],
        spec_460nm=reflectance[6], spec_470nm=reflectance[7], spec_480nm=reflectance[8],
        spec_490nm=reflectance[9], spec_500nm=reflectance[10], spec_510nm=reflectance[11],
        spec_520nm=reflectance[12], spec_530nm=reflectance[13], spec_540nm=reflectance[14],
        spec_550nm=reflectance[15], spec_560nm=reflectance[16], spec_570nm=reflectance[17],
        spec_580nm=reflectance[18], spec_590nm=reflectance[19], spec_600nm=reflectance[20],
        spec_610nm=reflectance[21], spec_620nm=reflectance[22], spec_630nm=reflectance[23],
        spec_640nm=reflectance[24], spec_650nm=reflectance[25], spec_660nm=reflectance[26],
        spec_670nm=reflectance[27], spec_680nm=reflectance[28], spec_690nm=reflectance[29],
        spec_700nm=reflectance[30])
    return spc


# 将分光反射率转化为lab颜色
def reflectance2lab(reflectance):
    spec = reflectance_to_spectral_color(reflectance)
    lab = convert_color(spec, LabColor)
    return lab


# 计算CIE1976色差
def cie1976_color_diff(reflectance1, reflectance2):
    lab1 = reflectance2lab(reflectance1)
    lab2 = reflectance2lab(reflectance2)
    color_diff = delta_e_cie1976(lab1, lab2)
    return color_diff


# 计算CIEDE2000色差(kL、kC、kH取1)
def ciede2000_color_diff(reflectance1, reflectance2):
    lab1 = reflectance2lab(reflectance1)
    lab2 = reflectance2lab(reflectance2)
    color_diff = delta_e_cie2000(lab1, lab2)
    return color_diff


# 使用km模型计算配方的分光反射率
# 注意：这里的concentrations使用的是 color_num*sample_num 的二维数组
def conc2ref_km(concentrations, background=info.white_solvent_reflectance,
                base_conc=info.base_concentration,
                base_color_num=info.base_color_num, base_ref=info.base_reflectance,
                ref_dim=info.reflectance_dim):
    init_conc_array = np.repeat(base_conc.reshape(base_color_num, 1), ref_dim).reshape(base_color_num, ref_dim)
    reflectance = np.zeros(ref_dim * concentrations.shape[0]).reshape(ref_dim, concentrations.shape[0])

    fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
    fst = ((np.ones_like(base_ref) - base_ref) ** 2 / (base_ref * 2) - fsb) / init_conc_array
    fss = np.zeros(31 * concentrations.shape[0]).reshape(31, concentrations.shape[0])
    for i in range(info.reflectance_dim):
        for j in range(info.base_color_num):
            fss[i, :] += concentrations[:, j] * fst[j, i]
        fss[i, :] += np.ones(concentrations.shape[0]) * fsb[i]

    reflectance = fss - ((fss + 1) ** 2 - 1) ** 0.5 + 1
    reflectance = reflectance.transpose()
    return reflectance


# -----------------------------------------------网络模型--------------------------------------------------
def MMD_multiscale(x, y):
    print('MMD_multiscale_x:',x)
    print('MMD_multiscale_y:',y)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    for a in [0.2, 0.5, 0.9, 1.3]:
        XX += a ** 2 * (a ** 2 + dxx) ** -1
        YY += a ** 2 * (a ** 2 + dyy) ** -1
        XY += a ** 2 * (a ** 2 + dxy) ** -1

    return torch.mean(XX + YY - 2. * XY)


def fit(input, target):
    print('fit_input:',input)
    print('fit_target:',target)
    return torch.mean((input - target) ** 2)


def non_nagative_attachment(base, lamb, x):
    return 1. / torch.clamp(torch.pow(base, lamb * x[x < 0]), min=0.001)


# -----------------------------------------------画图--------------------------------------------------
# 损失函数图
def plot_losses(losses, name):
    fig = plt.figure(figsize=(6, 6))
    losses = np.array(losses)
    # 正向传播损失
    ax1 = fig.add_subplot(211)
    ax1.plot(losses[0],'g')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('forward_loss')
    # 反向传播损失
    ax2 = fig.add_subplot(212)
    ax2.plot(losses[1], 'r')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('backward_loss')
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    # 保存图片
    plt.savefig('loss_dir/%s.png' % name)
    plt.close()


# 单x轴对应单y轴
# 注意：此处的y_arr为一个二维的list,也就是包含多条线
def plot_xy(x_arr, x_name, y_arr, y_legend_arr, y_name, fig_name, fig_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(y_arr)):
        ax.plot(x_arr, y_arr[i], label=y_legend_arr[i])
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    plt.savefig('%s/%s.png' % (fig_dir, fig_name))


# 单x轴对应双y轴,两个y轴在同一图中
def plot_xyy1(x_arr, x_name, y1_arr, y1_legend_arr, y1_name, y2_arr, y2_legend_arr, y2_name, fig_name, fig_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(y1_arr)):
        ax.plot(x_arr, y1_arr[i], label=y1_legend_arr[i])
    ax2 = ax.twinx()
    for i in range(len(y2_arr)):
        ax2.plot(x_arr, y2_arr[i], label=y2_legend_arr[i])
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y1_name)
    ax.set_ylabel(y2_name)
    plt.savefig('%s/%s.png' % (fig_dir, fig_name))


# 单x轴对应双y轴，两个y轴在不同图中
def plot_xyy2(x_arr, x_name, y1_arr, y1_legend_arr, y1_name, y2_arr, y2_legend_arr, y2_name, fig_name, fig_dir):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    for i in range(len(y1_arr)):
        ax1.plot(x_arr, y1_arr[i], label=y1_legend_arr[i])
    ax1.set_xlabel(x_name)
    ax1.set_ylabel(y1_name)
    ax2 = fig.add_subplot(212)
    for i in range(len(y2_arr)):
        ax2.plot(x_arr, y2_arr[i], label=y2_legend_arr[i])
    ax2.set_xlabel(x_name)
    ax2.set_ylable(y2_name)
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig('%s/%s.png' % (fig_dir, fig_name))