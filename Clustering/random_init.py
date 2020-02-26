import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Math_v3.Functions as mf
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans,MeanShift



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

def random_com(size):
    ans = []
    a = np.zeros(21)
    a[0] = 1
    a[1] = 1
    a[2] = 1
    for i in range(size):
        b = np.copy(a)
        np.random.shuffle(b)
        com = np.random.rand(21)
        ans.append(com*b)
    return np.array(ans)

def main():
    # data about km
    data_c = np.load('data_c.npy')
    data_p = np.load('data_p.npy').T
    # get K/S of base
    base_f_km = mf.math_model(data_p[0], 'km')
    F_km = get_F(data_p[1:], data_c, base_f_km, 'km')

    floor = [0,1,2,3,4,5,6,7,8,9,1000]
    for x in range(10):
        test_com = random_com(1)[0]
        test_ref = Mix(F_km, base_f_km, test_com, 'km')
        count = np.zeros(10).astype(int)
        coms = random_com(65536)
        for c in coms:
            r = Mix(F_km,base_f_km,c,'km')
            dif = mf.color_diff(r,test_ref)
            for i in range(10):
                if floor[i]<dif and dif<floor[i+1]:
                    count[i] += 1
        print(count)

main()