import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
import Math_v3.Functions as mf
import numpy as np
from collections import Counter
from tqdm import  tqdm
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

# 6 , 12 , 18, 21
filiters = np.array(
    [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
     [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
     [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
)

def Mix(compose,data_c,data_p):
    base_f = mf.math_model(data_p[0],'four_flux')
    F = np.zeros_like(base_f)
    for i in range(compose.size):
        df = (mf.math_model(data_p[i*3 + 2],'four_flux') - base_f)/ data_c[i][1]
        F += df * compose[i]
    return mf.i_math_model(F + base_f,'four_flux')

def correct_func(x,w):
    ans = 0
    j = 2
    for i in w:
        if type(x) == type(w):
            xx = np.array(x**j).reshape(x.size,1)
            ans += xx.dot(i.reshape(1,31))
        else:
            ans += i*x**j
        j+=1
    return ans

def get_dfs_KM(w,data_c,data_p):
    base_f = mf.math_model(data_p[0],'four_flux')
    dfs = []
    for i in range(data_c.shape[0]):
        df = (mf.math_model(data_p[i * 3 + 1],'four_flux') - base_f - correct_func(data_c[i][0], w)) / data_c[i][0]
        df += (mf.math_model(data_p[i * 3 + 2],'four_flux') - base_f - correct_func(data_c[i][1], w)) / data_c[i][1]
        df += (mf.math_model(data_p[i * 3 + 3],'four_flux') - base_f - correct_func(data_c[i][2], w)) / data_c[i][2]
        dfs.append(df/3)
    return np.array(dfs)

# use this !!
def corrected_Mix(compose,w,dfs,base_f,scale = 0.56):
    total_c = np.sum(compose,1)
    F = compose.dot(dfs)
    correct_F = F + correct_func(total_c,w)*scale + base_f
    return mf.i_math_model(correct_F,'four_flux')


def main():
    R0 = 0.048

    fil = 3

    scale = [0.515, 0.515, 0.52, 0.56]

    w_names = [
        'data_w_size6.npy',
        'data_w_size12.npy',
        'data_w_size18.npy',
        'data_w_size21_v3_no_r0.npy',
    ]

    # data_c 浓度数据，21种色浆 * 3次取点
    # data_p 分光反射率数据 size: (1 + 21 * 3) * 31 , 1为基底
    data_c = np.load('data_c.npy')
    data_p = np.load('data_p.npy')

    c, p = mf.data_filiter(filiters[fil], data_c, data_p)
    p = p.T - R0

    # 计算基底K/S
    base_f = mf.math_model(p[0],'four_flux')

    # 获取w,计算单位k/s值
    w = np.load(w_names[fil]).T
    dfs = get_dfs_KM(w, c, p)

    RealCompose_2 = np.load('RealCompose_2.npy')
    RealIngredient_2 = np.load('RealIngredint_2.npy')

    Composes = np.append(RealCompose, RealCompose_2, axis=0)
    Ingredients = np.append(RealIngredient, RealIngredient_2, axis=0)

    print('\n using : ', w_names[fil])

    Y1 = []
    Y2 = []
    Y3 = []
    #X_ = np.linspace(0,0.1,100)

    max = 6

    for com in range(Composes.shape[0]):
        compose = mf.data_filiter_(filiters[fil], Composes[com])
        if Counter(compose)[0] == 18:
            y = mf.color_diff(Ingredients[com], Mix(compose, c, p + R0))
            if y < max:
                Y1.append(y)
            y = mf.color_diff(Ingredients[com], corrected_Mix(np.array([compose]), w, dfs, base_f, scale[fil]) + R0)
            if y < max:
                Y2.append(y)
            y = mf.color_diff(Ingredients[com], Mix(compose, c, p) + R0)
            if y < max:
                Y3.append(y)


    plt.figure()

    X = np.arange(len(Y1))

    plt.scatter(X, Y1, label='four_flux')
    plt.plot(X, np.repeat(sum(Y1) / len(Y1), len(Y1)))
    plt.text(1, sum(Y1) / len(Y1), sum(Y1) / len(Y1))

    plt.scatter(X, Y3, marker='*', label='corrected_four_flux_1.0')
    plt.plot(X, np.repeat(sum(Y3) / len(Y3), len(Y3)))
    plt.text(1, sum(Y3) / len(Y3), sum(Y3) / len(Y3))

    plt.scatter(X, Y2, marker='x', label='corrected_four_flux_2.0')
    plt.plot(X, np.repeat(sum(Y2) / len(Y2), len(Y2)))
    plt.text(1, sum(Y2) / len(Y2), sum(Y2) / len(Y2))

    plt.xlabel('加料方案', fontproperties=font_set)
    plt.ylabel('色差', fontproperties=font_set)


    # plt.imshow(Y)

    # ignore ticks
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.show()

main()