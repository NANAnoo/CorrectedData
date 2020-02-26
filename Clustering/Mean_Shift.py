import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from km_model.info import base_color_num, reflectance_dim
import Math_v3.Functions as mf
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans,MeanShift
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# F： 各色浆单位浓度下的K/S， m*n， m:色浆数， n:波长数
# c： 配方的浓度向量
# Model： 配色模型的反函数
def Mix(F,base_f,c,Model):
    return mf.i_math_model(c.dot(F)+ base_f,Model)


# 获取原料色浆的F矩阵, F：各色浆在每个波长单位浓度下的K/S， m*n， m:色浆数， n:波长数
def get_F(p,c,base,Model):
    ff = mf.math_model(p,Model) - base
    F = []
    sample_size = c.shape[1]

    for i in range(c.shape[0]):
        F.append(ff[sample_size*i + int(sample_size/2)] / c[i][int(sample_size/2)])
    return np.array(F)

def density_function(x,u,v):
    n = len(x)
    sx = x - u
    sx = sx[:,np.newaxis]
    return np.exp(-(sx.T.dot(sx)/(2*v**2))) / (v*((2*np.pi)**n)**0.5)

def generate_test_sample(test_ref, N_sample, y_noise_scale, dim_x, dim_y, dim_z, dim_total):
    test_samp = np.tile(np.array(test_ref), N_sample).reshape(N_sample, reflectance_dim)
    test_samp = torch.tensor(test_samp, dtype=torch.float)
    test_samp += y_noise_scale * torch.randn(N_sample, reflectance_dim)
    rand_z  = torch.randn(N_sample, dim_z)
    test_samp = torch.cat([rand_z,  # zeros_noise_scale *
                           torch.zeros(N_sample, dim_total - dim_y - dim_z),
                           test_samp], dim=1)
    test_samp = test_samp.to(device)
    rand_z = rand_z.cpu().data.numpy()
    sz = []
    for z in rand_z:
        sz.append(density_function(z,0,1))
    return test_samp, np.squeeze(np.array(sz))[:,np.newaxis]

def predict(concentrations, reflectance, test_samp, model):
    # 使用inn预测配方
    predict_formula = model(test_samp, rev=True)[:, :base_color_num]
    predict_formula = predict_formula.cpu().data.numpy()
    # 假设涂料浓度小于一定值时，就不需要这种涂料
    predict_formula = np.where(predict_formula < 0.0, 0, predict_formula)

    return predict_formula

def main():

    # data about km
    data_c = np.load('data_c.npy')
    data_p = np.load('data_p.npy').T
    # get K/S of base
    base_f_km = mf.math_model(data_p[0], 'km')
    F_km = get_F(data_p[1:], data_c, base_f_km, 'km')
    #
    # shu = np.zeros(base_color_num)
    # shu[0] = 1
    # shu[1] = 1
    # shu[2] = 1
    # np.random.shuffle(shu)
    # random_com = np.random.rand(base_color_num) * shu
    # test_ref = Mix(F_km, base_f_km, random_com, 'km')

    data = np.load('data/data_01.npz')
    concentrations = torch.from_numpy(data['concentrations']).float()
    reflectance = torch.from_numpy(data['reflectance']).float()
    random_com = concentrations[42].cpu().data.numpy()
    test_ref = reflectance[42].cpu().data.numpy()


    # data about inn
    inn = torch.load('model_dir/model_02')
    y_noise_scale = 1e-4
    N_sample = 20000
    N_class = 100
    dim_x = base_color_num
    dim_y = reflectance_dim
    dim_z = 13
    dim_total = max(dim_x, dim_y + dim_z)

    sample,densi = generate_test_sample(test_ref, N_sample, y_noise_scale, dim_x, dim_y, dim_z, dim_total)
    p_c = predict(0,0,sample,inn)
    min = 100
    index = -1
    floor = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]
    count = np.zeros(10).astype(int)
    for i in range(N_sample):
        df = mf.color_diff(Mix(F_km,base_f_km,p_c[i],'km'),test_ref)
        for i in range(10):
            if floor[i] < df and df < floor[i + 1]:
                count[i] += 1
        if min>df:
            min = df
            index = i
    print(min)
    print(count)

    N_sample = 10000
    p_c = p_c[:N_sample][:]
    # p_c = p_c*densi / densi.max()
    kmeans_model = KMeans(N_class).fit(p_c)
    labels = kmeans_model.labels_

    # using center
    for c in kmeans_model.cluster_centers_:
        print('center_color_diff:',mf.color_diff(Mix(F_km,base_f_km,c,'km'),test_ref))

    p_c = np.concatenate((p_c, [random_com],kmeans_model.cluster_centers_))
    tsne = TSNE(n_components=2)
    tsne.fit_transform(p_c)
    res = np.array(tsne.embedding_)

    d = []
    for j in range(N_class):
        l = []
        for i in range(N_sample):
            if j == labels[i]:
                l.append(res[i])
        d.append(np.array(l).T)

    fig = plt.figure()
    # ax = Axes3D(fig)
    for j in range(N_class):
        plt.scatter(d[j][0],d[j][1],s=1)
    plt.scatter(res[index][0],res[index][1],marker='o',c='r',label ='min')
    plt.scatter(res[N_sample][0], res[N_sample][1], marker='v', c='b',label = 'real')
    # plot center
    for i in range(N_class):
        plt.scatter(res[N_sample+1 + i][0],res[N_sample+1 + i][1],marker='s',c='g')
    plt.legend()
    plt.show()

main()