import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
import Math_v3.Functions as mf
import numpy as np
from tqdm import tqdm

RealCompose = np.array(
    [
        [0.127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0748, 0, 0, 0, 0, 0, 0.56, 0, 0, 0],
        [0, 0.8014, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1491, 0, 0, 0, 0.2241, 0],
        [0.0451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7428, 0, 0, 0, 0, 0, 0, 0, 0.3364, 0],
        [0.3306, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0615, 0, 0.1219, 0, 0, 0]
    ]
)

RealIngredient = np.array(
    [
     [0.2869235,0.3562616,0.3648288,0.3690848,0.3720965,0.3743734,0.3748034,0.3726986,0.3670701,0.3595550,
      0.3495962,0.3348914,0.3152257,0.2898063,0.2588453,0.2237525,0.1902168,0.1687493,0.1594849,0.1573166,
      0.1543398,0.1508693,0.1542613,0.1652469,0.1817535,0.1943915,0.2001948,0.1974784,0.1753394,0.1711839,
      0.1716521],
     [0.2673378,0.3132285,0.3183329,0.3234908,0.3318701,0.3409707,0.3604081,0.4168356,0.5351773,0.6202191,
      0.6618687,0.6919741,0.7136238,0.7292901,0.7314631,0.7131701,0.6773048,0.6302681,0.5738088,0.5133060,
      0.4535525,0.4108878,0.3908512,0.3808001,0.3752591,0.3727644,0.3801365,0.3976869,0.4237110,0.4332685,
      0.4433292],
     [0.2805455,0.3224138,0.3143094,0.3027386,0.2937162,0.2832071,0.2719996,0.2643588,0.2582215,0.2541490,
      0.2498622,0.2446374,0.2448266,0.2456081,0.2388589,0.2233387,0.2128640,0.2172354,0.2586341,0.4077814,
      0.5150258,0.5389543,0.5433356,0.5433164,0.5431498,0.5423625,0.5413173,0.5409844,0.5348357,0.5314131,
      0.5256729],
     [0.2335778,0.2669207,0.2692738,0.2711587,0.2714446,0.2716927,0.2717375,0.2719162,0.2708953,0.2690464,
      0.2680669,0.2653399,0.2614551,0.2566476,0.2497642,0.2403944,0.2294249,0.2198784,0.2130321,0.2085395,
      0.2036385,0.1986061,0.1988262,0.2010616,0.2045918,0.2065673,0.2080161,0.2085742,0.2017922,0.2004079,
      0.1997598]
    ]
)


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

# E = (C dot F - tergetF)^2
#   = (C dot F - targetF).T dot (C dot F - targetF)
# dE(C) / d(C) = C dot ( F dot F.T ) - targetF dot F.T = 0
# C = targetF dot F.T dot [ ( F dot F.T )^-1 ]
# 预测配方
def PredictCompose(F,base_f,targetIngredint,Model):
    target_f = mf.math_model(targetIngredint,Model) - base_f
    return target_f.dot(F.T).dot(np.linalg.inv(F.dot(F.T)))


def main():

    Model = 'four_flux'

    R_km = 0
    R_muf = 0

    # load data
    RealCompose_2 = np.load('RealCompose_2.npy')
    RealIngredient_2 = np.load('RealIngredint_2.npy')

    Composes = np.append(RealCompose,RealCompose_2,axis=0)
    Ingredients = np.append(RealIngredient,RealIngredient_2,axis=0)

    data_c = np.load('data_c.npy')
    data_p = np.load('data_p.npy').T

    #get K/S of base
    base_f_km = mf.math_model(data_p[0] - R_km, 'km')
    base_f_muf = mf.math_model(data_p[0] - R_muf, 'four_flux')

    #get F
    F_km = get_F(data_p[1:] - R_km,data_c,base_f_km,'km')
    F_muf = get_F(data_p[1:] - R_muf,data_c,base_f_muf,'four_flux')

    diffs_km = []
    diffs_muf = []

    for i in range(Composes.shape[0]):
        PredictIngredint_km = Mix(F_km,base_f_km,Composes[i],'km') + R_km
        PredictIngredint_muf = Mix(F_muf, base_f_muf, Composes[i], 'four_flux') + R_muf
        dif = mf.color_diff(PredictIngredint_km,Ingredients[i])
        if dif < 6:
         diffs_km.append(dif)

        dif = mf.color_diff(PredictIngredint_muf, Ingredients[i])
        if dif < 6:
            diffs_muf.append(dif)

    plt.figure()

    X = np.arange(len(diffs_km))
    plt.scatter(X, diffs_km,label='KM')
    plt.plot(X, np.repeat(sum(diffs_km)/len(diffs_km),len(diffs_km)))
    plt.text(1,sum(diffs_km)/len(diffs_km),sum(diffs_km)/len(diffs_km))

    X = np.arange(len(diffs_muf))
    plt.scatter(X, diffs_muf,label='four_flux')
    plt.plot(X, np.repeat(sum(diffs_muf)/len(diffs_muf),len(diffs_muf)))
    plt.text(1,sum(diffs_muf)/len(diffs_muf),sum(diffs_muf)/len(diffs_muf))


    plt.xlabel('加料方案',fontproperties=font_set)
    plt.ylabel('色差',fontproperties=font_set)

    plt.legend()
    print(matplotlib.get_backend())
    plt.show()

    #
    # print(Model, 'compare REAL    ',mf.color_diff(PredictIngredint,RealIngredint[5]))
    #
    # p_com = PredictCompose(F,base_f,PredictIngredint,Model)
    # print(p_com)
    # print(RealCompose[0])
    # print('PreDict Color_diff ',mf.color_diff(PredictIngredint,Mix(F,base_f,p_com,Model)))
#
# def main():
#     RealCompose = np.load('RealCompose_2.npy')
#     data_c = np.load('data_c.npy')
#     data_p = np.load('data_p.npy').T
#
#     #get K/S of base
#     base_f = mf.math_model(data_p[0])
#
#     #get F
#     F = get_F(data_p[1:],data_c)
#
#     StandardIngredint = Mix(F, base_f, RealCompose[0])
#
#
#     sample_size = 1
#     test_size = 21
#     X , Y= [], []
#
#     for i in range(test_size):
#         best = 1000
#         X.append(sample_size)
#         for j in tqdm(range(sample_size)):
#             com = np.random.rand(RealCompose[0].size)
#             diff = mf.color_diff(Mix(F,base_f,com),StandardIngredint)
#             if diff < best:
#                 best = diff
#         Y.append(best)
#         sample_size *= 2
#
#     np.save('X',np.array(X))
#     np.save('Y', np.array(Y))

    # plt.figure()
    #
    # plt.plot(X,Y)
    #
    # plt.savefig('01.png')
    #
    # plt.show()



main()