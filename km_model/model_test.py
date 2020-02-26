import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time
from km_model.info import base_color_num, reflectance_dim
from km_model.utils import conc2ref_km, ciede2000_color_diff
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 预测配方
def predict(concentrations, reflectance, test_samp, model):
    # 使用inn预测配方
    predict_formula = model(test_samp, rev=True)[:, :base_color_num]
    predict_formula = predict_formula.cpu().data.numpy()
    # 假设涂料浓度小于一定值时，就不需要这种涂料
    predict_formula = np.where(predict_formula < 0.1, 0, predict_formula)

    return predict_formula


# 根据分光反射率预测出配方，对于每一个分光反射率，选择色差最小的三个配方
# concentrations:目标真实配方，reflectance:目标分光反射率，predict_formula:预测的配方
def test1(concentrations, reflectance, predict_formula):
    # 计算预测配方的反射率信息
    formula_ref = conc2ref_km(predict_formula)
    # 用于记录色差最小的三个配方
    top3 = [[100, 0], [100, 0], [100, 0]]
    for n in range(formula_ref.shape[0]):
        diff = ciede2000_color_diff(reflectance, formula_ref[n, :])
        if diff < top3[2][0]:
            top3[2][0] = diff
            top3[2][1] = n
            top3.sort()
    print('real_formula:')
    print(concentrations)
    for n in range(3):
        print('predict_formula_', n)
        print(predict_formula[top3[n][1], :])
        print("color diff: %.2f \n" % top3[n][0])
    print("\n\n")


# 选择满足色浆种类不超过5种，且色差最小的三个配方
# concentrations:目标真实配方，reflectance:目标分光反射率，predict_formula:预测的配方
def test2(concentrations, reflectance, predict_formula):
    # 计算预测配方的反射率信息
    formula_ref = conc2ref_km(predict_formula)
    # 用于记录色差最小的三个配方
    top3 = [[100, 0], [100, 0], [100, 0]]
    for n in range(formula_ref.shape[0]):
        diff = ciede2000_color_diff(reflectance, formula_ref[n, :])
        temp_formula = predict_formula[n, :].copy
        temp_formula[temp_formula != 0] = 1
        # 只选择使用的色浆种类小于等于5种的配方
        if (sum(temp_formula) <= 5 and diff < top3[2][0]):
            top3[2][0] = diff
            top3[2][1] = n
            top3.sort()
        print('real_formula:')
        print(concentrations)
        for n in range(3):
            print('predict_formula_', n)
            print(predict_formula[top3[n][1], :])
            print("color diff: %.2f \n" % top3[n][0])
        print("\n\n")


def main():
    # 加载模型
    inn = torch.load('model_dir/model_01')
    # 读取数据集
    data = np.load('data_dir/data_01.npz')
    concentrations = torch.from_numpy(data['concentrations']).float()
    reflectance = torch.from_numpy(data['reflectance']).float()
    # 加载数据
    testsplit = 3 * 3
    test_conc = concentrations[:testsplit]
    test_ref = reflectance[:testsplit]
    # 选取的样本数
    N_sample = 20000
    y_noise_scale = 3e-2
    dim_x = base_color_num
    dim_y = reflectance_dim
    dim_z = 13
    dim_total = max(dim_x, dim_y + dim_z)
    # 进行测试
    for i in range(testsplit):
        test_samp = np.tile(np.array(test_ref[i]), N_sample).reshape(N_sample, reflectance_dim)
        test_samp = torch.tensor(test_samp, dtype=torch.float)
        test_samp += y_noise_scale * torch.randn(N_sample, reflectance_dim)
        test_samp = torch.cat([torch.randn(N_sample, dim_z),  # zeros_noise_scale *
                               torch.zeros(N_sample, dim_total - dim_y - dim_z),
                               test_samp], dim=1)
        test_samp = test_samp.to(device)
        # 测试
        print('test sample:', i)
        predict_formula = predict(test_conc[i], test_ref[i], test_samp, inn)
        # test1(test_conc[i],test_ref[i],predict_formula)
        test1(test_conc[i], test_ref[i], predict_formula)


def plot():
    # 加载模型
    inn = torch.load('model_dir/model_02')
    # 读取数据集
    data = np.load('data_dir/data_01.npz')
    concentrations = torch.from_numpy(data['concentrations']).float()
    reflectance = torch.from_numpy(data['reflectance']).float()
    # 加载数据
    testsplit = 3
    test_conc = concentrations[:testsplit]
    test_ref = reflectance[:testsplit]
    # 选取的样本个数序列
    #sample_range = np.array([100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
    sample_range=np.array([100,200,300,400,500,600,700,800,900,1000])
    y_noise_scale = 1e-4
    dim_x = base_color_num
    dim_y = reflectance_dim
    dim_z = 13
    dim_total = max(dim_x, dim_y + dim_z)
    # 平均最小色差
    min_avg_arr = []
    # 平均预测时间
    time_avg_arr = []
    for i in range(testsplit):
        temp1 = []
        temp2 = []
        for n in tqdm(sample_range):
            N_sample = n
            min_arr = []
            time_arr = []
            for k in range(5):
                t_start = time()
                test_samp = generate_test_sample(test_ref[i], N_sample, y_noise_scale, dim_x, dim_y, dim_z, dim_total)
                predict_formula = predict(test_conc[i], test_ref[i], test_samp, inn)
                temp_diff = min_color_diff(test_conc[i], test_ref[i], predict_formula)
                min_arr.append(temp_diff)
                time_cost = time() - t_start
                time_arr.append(time_cost)
            min_avg = sum(min_arr) / 5
            time_avg = sum(time_arr) / 5
            temp1.append(min_avg)
            temp2.append(time_avg)
        min_avg_arr.append(temp1)
        time_avg_arr.append(temp2)
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(211)
    ax1.plot(sample_range, min_avg_arr[0],label='sample_01')
    ax1.plot(sample_range, min_avg_arr[1],label='sample_02')
    ax1.plot(sample_range, min_avg_arr[2],label='sample_03')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('sample_num')
    ax1.set_ylabel('color_diff')
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('sample_num')
    ax2.set_ylabel('time_cost')
    ax2.plot(sample_range, time_avg_arr[0],label='sample_01')
    ax2.plot(sample_range, time_avg_arr[1], label='sample_02')
    ax2.plot(sample_range, time_avg_arr[2], label='sample_03')
    ax2.legend(loc='upper right')
    #plt.show()
    plt.savefig('fig2.png')


# 生成测试样本：对配方和分光反射率数据做补充
def generate_test_sample(test_ref, N_sample, y_noise_scale, dim_x, dim_y, dim_z, dim_total):
    test_samp = np.tile(np.array(test_ref), N_sample).reshape(N_sample, reflectance_dim)
    test_samp = torch.tensor(test_samp, dtype=torch.float)
    test_samp += y_noise_scale * torch.randn(N_sample, reflectance_dim)
    test_samp = torch.cat([torch.randn(N_sample, dim_z),  # zeros_noise_scale *
                           torch.zeros(N_sample, dim_total - dim_y - dim_z),
                           test_samp], dim=1)
    test_samp = test_samp.to(device)
    return test_samp


# 计算预测配方的最小色差
def min_color_diff(concentrations, reflectance, predict_formula):
    # 计算预测配方的反射率信息
    formula_ref = conc2ref_km(predict_formula)
    # 用于最小色差
    min_diff = 100
    for n in range(formula_ref.shape[0]):
        diff = ciede2000_color_diff(reflectance, formula_ref[n, :])
        if diff < min_diff:
            min_diff = diff
    return min_diff


#main()
plot()
