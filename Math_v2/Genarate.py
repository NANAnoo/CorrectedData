import Math_v2.Functions as mf
import numpy as np

RealCompose = np.array(
    [   [0.127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0748, 0, 0, 0, 0, 0, 0.56, 0, 0, 0],
        [0, 0.8014, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1491, 0, 0, 0, 0.2241, 0],
        [0.0451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7428, 0, 0, 0, 0, 0, 0, 0, 0.3364, 0],
        [0.3306, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0615, 0, 0.1219, 0, 0, 0]
    ]
)

RealIngredient = np.array(
    [[0.2869235,0.3562616,0.3648288,0.3690848,0.3720965,0.3743734,0.3748034,0.3726986,0.3670701,0.3595550,
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

def Mix(compose,data_c,data_p):
    base_f = mf.math_model(data_p[0])
    F = np.zeros_like(base_f)
    for i in range(compose.size):
        df = (mf.math_model(data_p[i*3 + 2]) - base_f)/ data_c[i][1]
        F += df * compose[i]
    return mf.i_math_model(F + base_f)

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
    base_f = mf.math_model(data_p[0])
    dfs = []
    for i in range(data_c.shape[0]):
        df = (mf.math_model(data_p[i * 3 + 1]) - base_f - correct_func(data_c[i][0], w)) / data_c[i][0]
        df += (mf.math_model(data_p[i * 3 + 2]) - base_f - correct_func(data_c[i][1], w)) / data_c[i][1]
        df += (mf.math_model(data_p[i * 3 + 3]) - base_f - correct_func(data_c[i][2], w)) / data_c[i][2]
        dfs.append(df/3)
    return np.array(dfs)

# use this !!
def corrected_Mix(compose,w,dfs,base_f,scale = 0.56):
    total_c = np.sum(compose,1)
    F = compose.dot(dfs)
    return mf.i_math_model(F + correct_func(total_c,w)*scale + base_f)

def main():
    com = 1
    fil = 1

    scale = 0.56
    save_name = 'dataset_Corrected_size6'
    w_names = [
        'data_w_size6.npy',
        'data_w_size12.npy',
        'data_w_size18.npy',
        'data_w_size21.npy',
    ]
    # data_c 浓度数据，21种色浆 * 3次取点
    # data_p 分光反射率数据 size: (1 + 21 * 3) * 31 , 1为基底
    data_c = np.load('data_c.npy')
    data_p = np.load('data_p.npy')

    # 6 , 12 , 18, 21
    filiters = np.array(
        [[0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    )

    c, p = mf.data_filiter(filiters[fil], data_c, data_p)
    p = p.T
    compose = mf.data_filiter_(filiters[fil],RealCompose[com])

    base_f = mf.math_model(p[0])
    w = np.load(w_names[fil]).T
    dfs = get_dfs_KM(w,c,p)

    print(mf.color_diff(RealIngredient[com],corrected_Mix(np.array([compose]),w,dfs,base_f,scale)))
    print(mf.color_diff(RealIngredient[com],Mix(compose,c,p)))

main()