import sys
sys.path.append('../')
import time
import numpy as np
import km_model.info as info
import torch as torch
from itertools import combinations


# 用于生成配方，即色浆的浓度和对应的分光反射率
# total_data_size：总数据集大小
# seed: 随机数种子
def generate(total_data_size, prior_bound=[0, 1], seed=0, model='km'):
    np.random.seed(seed)

    # 从info中获取相关信息
    background = info.white_solvent_reflectance
    color_num = info.base_color_num
    chosen_color_num = info.chosen_color_num
    base_concentration = info.base_concentration
    base_reflectance = info.base_reflectance
    reflectance_dim = info.reflectance_dim

    # 在0-1的均匀分布中随机采样，生成total_data_size条数据，每条数据包含color_num种色浆的浓度
    # 放缩到prior_bound范围下
    concentrations = np.random.uniform(0, 1, size=(total_data_size, color_num))
    for i in range(color_num):
        concentrations[:, i] = prior_bound[0] + (prior_bound[1] - prior_bound[0]) * concentrations[:, i]

    # 在color_num种选取chosen_color_num种，其余的色浆浓度都置为0
    combine_list = list(combinations(np.arange(0, color_num, 1), color_num - chosen_color_num))
    combine_num = combine_list.__len__()
    n = total_data_size // combine_num  # 每种排列组合的配方个数
    for i in range(combine_num - 1):
        concentrations[i * n:(i + 1) * n, combine_list[i]] = 0.
    concentrations[(combine_num - 1) * n:, combine_list[combine_num - 1]] = 0.

    # 使用km模型生成分光反射率
    if model == 'km':
        # 原本为color_num*1，重复reflectance_dim次，再转变为color_num*reflectance_dim
        base_concentration_array = np.repeat(base_concentration.reshape(color_num, 1), reflectance_dim).reshape(
            color_num, reflectance_dim)
        # 基底的K/S值，论文公式4-6a
        fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
        # 各种色浆的单位K/S值，论文公式4-6b
        fst = ((np.ones_like(base_reflectance) - base_reflectance) ** 2 / (
                base_reflectance * 2) - fsb) / base_concentration_array
        # ydim*N的0
        fss = np.zeros(total_data_size * reflectance_dim).reshape(reflectance_dim, total_data_size)
        # 涂料的K/S值,论文公式4-6c
        for i in range(reflectance_dim):
            for j in range(color_num):
                fss[i, :] += concentrations[:, j] * fst[j, i]
            fss[i, :] += np.ones(total_data_size) * fsb[i]
        # 涂料的分光反射率，论文公式4-6d
        reflectance = fss - ((fss + 1) ** 2 - 1) ** 0.5 + 1
        # 转置
        reflectance = reflectance.transpose()
    else:
        print('Model has not been implemented')
        exit(1)

    # 对数据进行打乱
    shuffling = np.random.permutation(total_data_size)
    concentrations = torch.tensor(concentrations[shuffling], dtype=torch.float)
    reflectance = torch.tensor(reflectance[shuffling], dtype=torch.float)

    return concentrations, reflectance


def main():
    total_data_size = 2**25
    start = time.time()
    concentrations, reflectance = generate(total_data_size=total_data_size,prior_bound=[0,1])
    print('time-cost',(time.time() - start)/60)
    for con in concentrations:
        print(con)
    np.savez('data_dir\data_02', concentrations=concentrations, reflectance=reflectance)
    # np.savetxt('dataTxt', concentrations)


main()
