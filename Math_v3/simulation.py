import  Math_v3.Functions as mf
import numpy as np
from  tqdm import tqdm
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

data_c = np.load('data_c.npy')
data_p = np.load('data_p.npy').T

#get K/S of base
base_f_km = mf.math_model(data_p[0], 'km')

#get F
F_km = get_F(data_p[1:],data_c,base_f_km,'km')

com0 = np.zeros(21).astype(float)

Noise = [0.00001,0.0001,0.0005,0.001,0.005,0.01,0.02,0.03]
size = 50000

for noise in Noise:
    diff = 0
    for i in range(size):
        R0 = np.random.rand(31)
        diff += mf.color_diff(R0,R0 + noise * np.random.normal(0,1,31))
    print('mean_diff_with_noise_',noise,':',diff / size)